from __future__ import annotations
import os
from openai import OpenAI
from models import IntentResult

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPT_INTENT_PATH = os.path.join(BASE_DIR, "resource", "prompt_intent_estimate.txt")
with open(PROMPT_INTENT_PATH, "r", encoding="utf-8") as fp:
    PROMPT_TEMPLATE = fp.read().strip()


class IntentEstimator:
    """Estimate customer's primary intent from a transcript."""

    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini") -> None:
        self.client = openai_client
        self.model = model

    def estimate(self, transcript: str) -> IntentResult:
        """Return the predicted intent as IntentResult."""
        print(f"[IntentEstimator] Estimating intent...")
        prompt = PROMPT_TEMPLATE + "\n\n" + transcript

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=60,
        )

        intent_sentence = response.choices[0].message.content.strip()

        # Since confidence is not obtained from OpenAI, set it to None. Implement separately if needed.
        return IntentResult(intent=intent_sentence)
