from typing import Optional, List
from pydantic import BaseModel
from enum import Enum

# Enum for sales conversation phase
class PhaseLabel(str, Enum):
    introduction = "At the time of introduction"
    product_details = "Hearing about product details"
    price_negotiations = "At the time of price negotiations"
    schedule_adjustments = "At the time of schedule adjustments"
    before_purchase = "Just before purchase"
    after_closing = "After closing"
    other = "Other (Use only if not applicable to the above)"

PHASE_ENUMS = [e.value for e in PhaseLabel]

class Transcript(BaseModel):
    text: str

class QAResult(BaseModel):
    question: str
    customer_emotion: str
    phase: PhaseLabel

class ProductInfo(BaseModel):
    product_description: str
    brand: str = "unknown"
    desired_price: Optional[float] = None
    quoted_price: Optional[float] = None
    condition: str = "unspecified"
    intent: str = "undetermined"
    category: str = "その他"
    age_of_product: Optional[str] = None
    description: Optional[str] = None

class IntentResult(BaseModel):
    """Predicted customer intent sentence and optional confidence."""
    intent: str
    confidence: Optional[float] = None  # 0.0–1.0 range

class ProcessedAudioData(BaseModel):
    file_name: str
    questions: List[QAResult]
    product_info: List[ProductInfo]
    summary: Optional[str] = None
    
class AudioProcessingRequest(BaseModel):
    s3_bucket: str
    s3_prefix: str
    region: Optional[str] = "ap-northeast-1"

class V1QuestionRequest(BaseModel):
    product_info: List[ProductInfo]
    # intent: IntentResult
    phase: PhaseLabel
    n_questions: int = 5
    transcript: str

class V1QuestionResponse(BaseModel):
    questions: List[str]
    
class V1ExtractRequest(BaseModel):
    transcript: str

class V1ExtractResponse(BaseModel):
    product_info: List[ProductInfo]
    # intent: IntentResult
    phase: PhaseLabel