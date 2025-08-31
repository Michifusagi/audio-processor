import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import boto3
from dotenv import load_dotenv
from models import (
    AudioProcessingRequest,
    V1QuestionRequest,
    V1QuestionResponse,
    V1ExtractRequest,
    V1ExtractResponse
)
from services.audio_analysis_pipeline import AudioProcessor
from services.followup_question_generator import V1QuestionGenerator
from managers.product_and_phase_extractor import V1RequestExtractor

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Audio Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# -------------------------------------------
# Audio Processing Endpoint
# -------------------------------------------

@app.post("/process-audio")
def process_audio(
    request: AudioProcessingRequest,
    background_tasks: BackgroundTasks
):
    s3_client = boto3.client("s3", region_name=request.region)
    
    processor = AudioProcessor(
        s3_client=s3_client,
        s3_bucket=request.s3_bucket,
        s3_prefix=request.s3_prefix,
        openai_client=client
    )
    background_tasks.add_task(processor.run)
    return {"status": "Processing started in background."}

# -------------------------------------------
# V1 Question Generation Endpoint
# -------------------------------------------

qg = V1QuestionGenerator(openai_client=client)

@app.post("/v1/generate_questions", response_model=V1QuestionResponse)
async def generate_questions(
    request: V1QuestionRequest
) -> V1QuestionResponse:
    """
    Generate follow-up questions from product info & conversation phase.
    """
    try:
        result = qg.generate(
            product_info=request.product_info,
            phase=request.phase,
            n_questions=request.n_questions,
            transcript=request.transcript,
            # intent=request.intent,
        )
        return V1QuestionResponse(questions=result.questions)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# -------------------------------------------
# V1 Request Extraction Endpoint
# -------------------------------------------

v1_extractor = V1RequestExtractor(openai_client=client)

@app.post("/v1/extract_request", response_model=V1ExtractResponse)
async def extract_v1_request(request: V1ExtractRequest) -> V1ExtractResponse:
    """
    Extract product info and intent from transcript for V1 request.
    """
    try:
        result = v1_extractor.extract(request.transcript)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# -------------------------------------------
# Health Check (optional)
# -------------------------------------------

@app.get("/health", tags=["meta"])
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)