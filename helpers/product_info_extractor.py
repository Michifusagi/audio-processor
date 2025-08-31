from typing import List, Dict, Any
from openai import OpenAI
import json
import os
from helpers.scraper import AmazonScraper
from rapidfuzz import process, fuzz
from models import ProductInfo


class ProductInfoExtractor:
    """
    Extracts product information from customer conversation text,
    combines it with hierarchical category classification and external information retrieval,
    and returns it in a specified JSON structure.
    """
    
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    CATEGORY_PATH = os.path.join(BASE_DIR, "resource", "category_list.json")
    SCHEMA_PATH = os.path.join(BASE_DIR, "resource", "function_schema.json")
    PROMPT_PRODUCT_INFO_PATH = os.path.join(BASE_DIR, "resource", "prompt_product_info.txt")
    PROMPT_CATEGORY_PATH = os.path.join(BASE_DIR, "resource", "prompt_category.txt")

    # JSONを読み込み
    with open(CATEGORY_PATH, "r", encoding="utf-8") as f:
        CATEGORY_LIST = json.load(f)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        FUNCTION_SCHEMA = json.load(f)

    with open(PROMPT_PRODUCT_INFO_PATH, "r", encoding="utf-8") as f:
        PROMPT_PRODUCT_INFO = f.read()

    with open(PROMPT_CATEGORY_PATH, "r", encoding="utf-8") as f:
        PROMPT_CATEGORY = f.read()

    def __init__(self, openai_client: OpenAI, db_url, model: str="gpt-4o-mini"):
        self.client = openai_client
        self.model = model
        self.category_paths = self._extract_category_paths(self.CATEGORY_LIST)
        self.scraper = AmazonScraper()

    def extract(self, conversation_text: str) -> List[ProductInfo]:
        """
        Extracts product information from conversation text (excluding category, market_price, product_web_description)
        Then, supplements category, market_price, product_web_description and integrates them
        """
        # 1. Extract basic product information using LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.PROMPT_PRODUCT_INFO},
                {"role": "user", "content": conversation_text}
            ],
            tools=[
                {
                    "type": "function",
                    "function": self._product_info_function_schema()
                }
            ],
            tool_choice={"type": "function", "function": {"name": "extract_product_info"}}
        )

        # Extract function_call arguments from OpenAI response
        function_args = response.choices[0].message.tool_calls[0].function.arguments
        product_list_raw = json.loads(function_args)

        # Ensure the extracted data is always treated as a list
        if isinstance(product_list_raw, dict):
            product_list_raw = [product_list_raw]
        elif not isinstance(product_list_raw, list):
            raise ValueError("Unexpected format: Expected dict or list of dicts.")

        # Convert raw dict data into strongly-typed Pydantic models
        product_list = [ProductInfo(**prod) for prod in product_list_raw]

        results: list[ProductInfo] = []

        for prod in product_list:
            brand = prod.brand or "unknown"

            # Perform category classification based on product description
            category = self.classify_category(prod.product_description)

            # Retrieve product web description from external sources
            desc = self.scraper.get_market_price_and_description(prod.product_description)

            # Create a fully completed ProductInfo object with all supplemented fields
            prod_complete = ProductInfo(
                brand=brand,
                desired_price=prod.desired_price,
                quoted_price=prod.quoted_price,
                condition=prod.condition or "unspecified",
                intent=prod.intent or "undetermined",
                category=category,
                product_description=prod.product_description,
                age_of_product=prod.age_of_product,
                description=desc
            )

            results.append(prod_complete)

        return results


    def classify_category(self, product_description: str) -> str:
        """
        Based on product_description, output the category hierarchy path from CATEGORY_LIST using LLM
        """
        category_list_str = json.dumps(self.CATEGORY_LIST, ensure_ascii=False)
        prompt = (
            f"{self.PROMPT_CATEGORY}\n"
            f"商品説明: {product_description}\n"
            f"カテゴリリスト: {category_list_str}\n"
            "もっとも適切なカテゴリ階層パス（例: 家電 > キッチン家電）を日本語で1つだけ出力してください。"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_output = response.choices[0].message.content.strip()
        
        return self.normalize_category_path(raw_output)

    def _product_info_function_schema(self) -> Dict[str, Any]:
        """
        Returns a schema excluding category, market_price, product_web_description for Function Calling
        """
        import copy
        schema = copy.deepcopy(self.FUNCTION_SCHEMA)
        del schema["parameters"]["properties"]["category"]
        del schema["parameters"]["properties"]["market_price"]
        del schema["parameters"]["properties"]["product_web_description"]
        schema["parameters"]["required"] = [r for r in schema["parameters"]["required"]
                                            if r not in ["category", "market_price", "product_web_description"]]
        return schema
    
    
    def normalize_category_path(self, llm_output: str, threshold: int = 85) -> str:
        match, score, _ = process.extractOne(llm_output, self.category_paths, scorer=fuzz.token_sort_ratio)
        return match if score >= threshold else llm_output
    
    def _extract_category_paths(self, category_list, parent_path=""):
        paths = []
        for cat in category_list:
            current_path = f"{parent_path} > {cat['name']}" if parent_path else cat['name']
            paths.append(current_path)
            if "children" in cat:
                paths.extend(self._extract_category_paths(cat["children"], current_path))
        return paths