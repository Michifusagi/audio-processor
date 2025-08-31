from dotenv import load_dotenv
import os, re, requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class AmazonScraper:
    def __init__(self, threshold: int = 1):
        load_dotenv(override=True)
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.zenrows_key = os.getenv("ZENROWS_KEY")

        self.threshold = threshold
        self.session = requests.Session()
        # Add retry logic for more robust external API calls
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        })

    # ① Extract product name (as is in Japanese)
    def extract_product_description(self, user_input: str) -> str:
        return user_input.strip()

    # ② Description filter (always True for now)
    def is_relevant_description(self, *_):  # Keep type while disabling
        return True

    # ③ Get /dp/ URL with SerpApi
    def get_top_amazon_url(self, product_description: str) -> str:
        if not product_description:
            print("[WARN] Empty product description — skipping SerpApi request.")
            return ""

        try:
            params = {
                "engine": "amazon",
                "k": product_description,
                "amazon_domain": "amazon.co.jp",
                "api_key": self.serpapi_key,
                "page": "1"
            }
            r = self.session.get("https://serpapi.com/search", params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            results = data.get("shopping_results") or data.get("organic_results") or []
            for item in results:
                link = item.get("link", "")
                if "/dp/" in link:
                    return link
        except Exception as e:
            print(f"[ERROR] SerpApi: {e}")
        return ""

    # ④ Extract price and description with ZenRows + BeautifulSoup
    def fetch_product_detail(self, product_url: str):
        try:
            url = product_url
            if "language=" not in url:
                join_char = "&" if "?" in url else "?"
                url += f"{join_char}language=ja_JP"
                
            params = {
                "apikey": self.zenrows_key,
                "url": url,
                "js_render": "true",
                "premium_proxy": "true"
            }
            r = self.session.get("https://api.zenrows.com/v1/", params=params, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            desc = ""
            feature = soup.select_one("#feature-bullets")
            if feature:
                lis = feature.find_all("li")
                desc = " / ".join(li.get_text(strip=True) for li in lis if li.get_text(strip=True))
            if not desc:
                for sel in ("#productDescription p", "#bookDescription_feature_div"):
                    t = soup.select_one(sel)
                    if t and t.get_text(strip=True):
                        desc = t.get_text(strip=True)
                        break

            desc = re.sub(r"\s+", " ", desc).strip()
            return desc
        except Exception as e:
            print(f"[ERROR] ZenRows: {e}")
            return ""

    # ⑤ External API 
    def get_market_price_and_description(self, user_input: str):
        product_url = self.get_top_amazon_url(self.extract_product_description(user_input))
        if not product_url:
            print("[INFO] Amazon product page not found")
            return ""

        desc = self.fetch_product_detail(product_url)
        if not desc:
            print("[INFO] Could not retrieve description")
            return ""

        return desc