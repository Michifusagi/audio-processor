from __future__ import annotations
from helpers.product_info_extractor import ProductInfoExtractor
from helpers.intent_estimator import IntentEstimator
from typing import List
from models import ProductInfo, IntentResult, V1ExtractResponse, PHASE_ENUMS
import os

class V1RequestExtractor:
    
    def __init__(self, openai_client) -> None:
        self.openai_client = openai_client
        db_url = os.getenv("DATABASE_URL")
        self.product_info_extractor = ProductInfoExtractor(openai_client=self.openai_client, db_url=db_url)
        self.intent_estimator = IntentEstimator(openai_client=self.openai_client)
        
    def extract(self, transcript: str) -> V1ExtractResponse:
        """Extract product info and intent from transcript."""

        product_info: List[ProductInfo] = self.product_info_extractor.extract(transcript)
        # intent_result: IntentResult = self.intent_estimator.estimate(transcript)
        # intent_result = IntentResult(
        #     intent="undetermined",  # Default value, can be updated based on actual intent estimation
        #     confidence=None  # Optional confidence, can be set if available
        # )

        # Extract phase (get the final phase at the end of the conversation using function calling)
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        PROMPT_PATH = os.path.join(BASE_DIR, "resource", "prompt_final_phase.txt")
        with open(PROMPT_PATH, "r", encoding="utf-8") as fp:
            prompt_template = fp.read().strip()
        prompt = prompt_template.format(TRANSCRIPT=transcript)

        phase_function = {
            "type": "function",
            "function": {
                "name": "record_final_phase",
                "description": "会話終了時点での営業プロセスのphaseラベルを記録する",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phase": {"type": "string", "enum": PHASE_ENUMS}
                    },
                    "required": ["phase"]
                }
            }
        }
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            tools=[phase_function],
            tool_choice={"type": "function", "function": {"name": "record_final_phase"}},
            temperature=0.2,
            max_tokens=20,
        )
        phase = response.choices[0].message.tool_calls[0].function.arguments
        import json
        phase_dict = json.loads(phase)
        phase_label = phase_dict["phase"]

        return V1ExtractResponse(
            product_info=product_info,
            # intent=intent_result,
            phase=phase_label
        )
