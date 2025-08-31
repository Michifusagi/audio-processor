from __future__ import annotations
from typing import List
from openai import OpenAI
import json
from models import QAResult
import os
from pydantic import ValidationError

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPT_Q_PATH = os.path.join(BASE_DIR, "resource", "prompt_question_extraction.txt")
with open(PROMPT_Q_PATH, "r", encoding="utf-8") as fp:
    PROMPT_TEMPLATE = fp.read().strip()

PHASE_ENUMS = [
    "At the time of introduction",
    "Hearing about product details",
    "At the time of price negotiations",
    "At the time of schedule adjustments",
    "Just before purchase",
    "After closing",
    "Other (Use only if not applicable to the above)"
]

def normalize_phase(phase_value):
    if not isinstance(phase_value, str):
        return None
    # Exact match
    if phase_value in PHASE_ENUMS:
        return phase_value
    # If contains "Other"
    if "Other" in phase_value:
        return "Other (Use only if not applicable to the above)"
    # Partial match (return the closest one)
    for enum in PHASE_ENUMS:
        if phase_value.strip() in enum:
            return enum
    # Exact match after stripping whitespace
    for enum in PHASE_ENUMS:
        if phase_value.strip() == enum:
            return enum
    return None  # If cannot be normalized

class QuestionExtractor:
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "record_qa_result",
                    "description": "Record one sales question and customer sentiment/phase",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question":         {"type": "string"},
                            "customer_emotion": {"type": "string",
                                                "enum": ["positive", "neutral",
                                                        "uncertain", "negative"]},
                            "phase":            {"type": "string",
                                                "enum": PHASE_ENUMS}
                        },
                        "required": ["question", "customer_emotion", "phase"]
                    }
                }
            }
        ]

    def extract(self, text: str) -> List[QAResult]:
        msgs = [
            {"role": "system", "content": "あなたは優秀な営業トレーナーです。"},
            {"role": "user", "content": self._build_prompt(text)}
        ]

        results: List[QAResult] = []

        max_loops = 10  # Prevent infinite loop
        for loop_count in range(max_loops):
            res = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.4,
            )
            msg = res.choices[0].message
            msgs.append(msg.model_dump(exclude_unset=True))  # Keep assistant message

            if not msg.tool_calls:       # No more questions → exit loop
                break

            for tc in msg.tool_calls:    # Parse each tool call
                payload = json.loads(tc.function.arguments)
                phase = payload.get("phase")
                normalized = normalize_phase(phase)
                if normalized:
                    payload["phase"] = normalized
                    try:
                        results.append(QAResult(**payload))
                    except ValidationError as e2:
                        print("Validation error after normalization:", e2)
                else:
                    print(f"Unrecognized phase value: {phase}")
                # Acknowledge tool completion
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "OK",
                    }
                )
        else:
            print("[QuestionExtractor] Warning: loop limit reached in extract()")

        return results

    def _build_prompt(self, transcript: str) -> str:
            """Fill {TRANSCRIPT} placeholder in the prompt template."""
            return PROMPT_TEMPLATE.format(TRANSCRIPT=transcript)