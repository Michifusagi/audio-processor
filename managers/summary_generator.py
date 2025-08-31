from __future__ import annotations
from openai import OpenAI
import os

class SummaryGenerator:
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    PROMPT_PATH = os.path.join(BASE_DIR, "resource", "prompt_summary.txt")

    with open(PROMPT_PATH, "r", encoding="utf-8") as fp:
        PROMPT_TEMPLATE = fp.read().strip()

    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model

    def generate(self, transcript: str) -> str:
        messages = [
            {"role": "system", "content": "あなたは優秀な営業トレーナーです。"},
            {"role": "user", "content": self._build_prompt(transcript)}
        ]

        res = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4
        )

        return res.choices[0].message.content.strip()

    def _build_prompt(self, transcript: str) -> str:
        return self.PROMPT_TEMPLATE.format(TRANSCRIPT=transcript)