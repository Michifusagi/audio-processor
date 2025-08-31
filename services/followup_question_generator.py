# services/question_generator.py
from __future__ import annotations
import os
import re
from typing import List
from openai import OpenAI
from models import V1QuestionResponse, ProductInfo, PhaseLabel
from managers.summary_generator import SummaryGenerator
from helpers.vectordb import VectorStoreHandler


class V1QuestionGenerator:
    """Generate purchase promotion questions from multiple product info items and user intent."""
    
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    PROMPT_Q_PATH = os.path.join(BASE_DIR, "resource", "prompt_question_generation.txt")
    with open(PROMPT_Q_PATH, "r", encoding="utf-8") as fp:
        PROMPT_TEMPLATE = fp.read().strip()

    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini") -> None:
        self.client = openai_client
        self.model = model
        self.system_prompt = self.PROMPT_TEMPLATE
        self.summary_generator = SummaryGenerator(openai_client=self.client, model=self.model)
        self.vector_handler = VectorStoreHandler()

    def generate(
        self,
        product_info: List[ProductInfo] | str,
        phase: PhaseLabel | str,
        transcript: str,
        *,
        n_questions: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> V1QuestionResponse:
        """
        Generate follow-up questions using the following flow:
        1. Generate summary from transcript using SummaryGenerator
        2. Combine product_info and summary into a single text block
        3. Retrieve ret_questions by vector search (passing phase as well)
        4. Generate new questions using OpenAI with ret_questions, phase, and summary
        5. Return questions only
        """
        # 1. Generate summary from transcript
        summary = self.summary_generator.generate(transcript)

        # 2. Combine product_info and summary
        if isinstance(product_info, str):
            combined_summary = product_info + "\n" + summary
        else:
            items_block = "[Items for Purchase】\n"
            for p in product_info:
                items_block += f"**Name:** {p.product_description}\n"
                items_block += f"**Brand:** {p.brand}\n"
                items_block += f"**Quoted Price:** {p.quoted_price if hasattr(p, 'quoted_price') and p.quoted_price is not None else ''}\n"
                items_block += f"**Description:** {p.description if hasattr(p, 'description') and p.description else ''}\n\n"
            combined_summary = items_block + (summary or "")

        # 3. Retrieve ret_questions by vector search (passing phase as well)
        ret_questions = self.vector_handler.search_similar_questions(combined_summary, phase)
        
        print("ret_questions", ret_questions)

        # 4. Generate new questions using OpenAI with ret_questions, phase, and summary
        phase_label = phase.value if isinstance(phase, PhaseLabel) else str(phase)
        ret_questions_block = "\n".join(ret_questions)
        user_prompt = (
            f"- 参考質問\n{ret_questions_block}\n\n"
            f"- 会話のフェーズ\n{phase_label}\n\n"
            f"- 会話要約\n{summary}\n\n"
            f"上記を参考に、現在話題に上がっている商品以外についての買取促進の質問を {n_questions} 個出力してください。\n"
            f"特に「会話のフェーズ」を十分に考慮し、そのフェーズに最適な質問を考えてください。\n"
            f"質問文のみを自然な口調で、1 行につき 1 つずつ出力してください。番号や記号は不要です。"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        raw_text = content.strip() if content is not None else ""
        questions = self._parse_questions(raw_text, n_questions)
        return V1QuestionResponse(questions=questions)

    @staticmethod
    def _format_product_info(product_info: List[ProductInfo] | str) -> str:
        """Format detailed product info list as a readable text block"""
        if isinstance(product_info, str):
            return product_info

        rows: list[str] = []
        for idx, p in enumerate(product_info, 1):
            rows.append(
                f"{idx}. 商品説明: {p.product_description or '（なし）'}\n"
                f"   ブランド: {p.brand or '不明'}\n"
                f"   希望価格: {p.desired_price if p.desired_price is not None else '不明'}\n"
                f"   提示価格: {p.quoted_price if p.quoted_price is not None else '不明'}\n"
                f"   状態: {p.condition or '不明'}\n"
                f"   カテゴリ: {p.category or '不明'}\n"
                f"   製品の年数: {p.age_of_product if p.age_of_product is not None else '不明'}\n"
                f"   Web説明: {p.description or '（なし）'}"
            )
        return "\n\n".join(rows)

    @staticmethod
    def _parse_questions(raw: str, n: int) -> list[str]:
        """
        Extract clean questions from LLM output.
        Strips bullet/number prefixes like '1. ', '- ', '• ', etc.
        """
        lines = raw.splitlines()
        questions: list[str] = []
        for line in lines:
            cleaned = re.sub(r"^(\d+[\.\)]|[-•●①-⑩])\s*", "", line).strip()
            if cleaned:
                questions.append(cleaned)
            if len(questions) >= n:
                break
        return questions
