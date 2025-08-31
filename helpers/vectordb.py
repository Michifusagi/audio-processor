from __future__ import annotations

import os
import uuid
from typing import Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
import json
import traceback


class VectorStoreHandler:
    """Handle insertion of processed call-analysis data into Qdrant."""

    DEFAULT_COLLECTION_NAME = "market-enterprise"  # <- fixed internally
    DEFAULT_DIMENSION = 1536
    DEFAULT_DISTANCE = Distance.COSINE

    def __init__(self) -> None:
        # Load environment variables
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY", None)
        
        # Initialize Qdrant client
        self.client = QdrantClient(url=url, api_key=api_key)

        # Create collection if not exists
        collections = self.client.get_collections().collections
        if self.DEFAULT_COLLECTION_NAME not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=self.DEFAULT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.DEFAULT_DIMENSION,
                    distance=self.DEFAULT_DISTANCE,
                ),
            )
            # Also create an index for the file_name field (use create_payload_index in v1.14.3)
            self.client.create_payload_index(
                collection_name=self.DEFAULT_COLLECTION_NAME,
                field_name="file_name",
                field_schema="keyword"
            )
        else:
            # If the file_name index does not exist in an existing collection, create it
            collection_info = self.client.get_collection(self.DEFAULT_COLLECTION_NAME)
            field_indexes = getattr(collection_info, 'payload_schema', {})
            if 'file_name' not in field_indexes:
                self.client.create_payload_index(
                    collection_name=self.DEFAULT_COLLECTION_NAME,
                    field_name="file_name",
                    field_schema="keyword"
                )

        # Connect to collection via llama_index
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.DEFAULT_COLLECTION_NAME,
        )
        self.embed_model = OpenAIEmbedding()

    def store_from_processed_data(self, processed_data: Dict[str, Any]) -> None:
        """Insert a single embedded summary+product_info record into Qdrant."""
        summary = processed_data.summary
        product_info = processed_data.product_info or []
        file_name = processed_data.file_name if hasattr(processed_data, 'file_name') else None

        # Combine product_info and summary into one text block
        items_block = "[Items for Purchase】\n"
        for p in product_info:
            items_block += f"**Name:** {p.product_description}\n"
            items_block += f"**Brand:** {p.brand}\n"
            items_block += f"**Quoted Price:** {p.quoted_price if hasattr(p, 'quoted_price') and p.quoted_price is not None else ''}\n"
            items_block += f"**Description:** {p.description if hasattr(p, 'description') and p.description else ''}\n\n"
        text_to_embed = items_block + (summary or "")

        # Only one node per summary
        metadata = {
            'questions': json.dumps([q.model_dump() for q in processed_data.questions]),
            'file_name': file_name,
        }

        node = TextNode(
            id_=str(uuid.uuid4()),
            text=text_to_embed,
            metadata=metadata,
        )
        nodes = [node]

        # Push to Qdrant
        if nodes:
            embeddings = self.embed_model.get_text_embedding_batch([n.text for n in nodes])

            for n, e in zip(nodes, embeddings):
                n.embedding = e

            self.vector_store.add(nodes)
        else:
            print("[INFO] No summary found; nothing was indexed.")

    def exists_by_filename(self, file_name: str) -> bool:
        """Check if a vector with the given file_name exists in Qdrant."""
        result = self.client.scroll(
            collection_name=self.DEFAULT_COLLECTION_NAME,
            scroll_filter={
                "must": [
                    {"key": "file_name", "match": {"value": file_name}}
                ]
            },
            limit=1
        )
        # Support both tuple and dict return types
        points = result[0] if isinstance(result, tuple) else result.get("points", [])
        return len(points) > 0

    def search_similar_questions(self, summary: str, phase) -> list[str]:
        """
        Given summary and phase, search Qdrant for similar nodes and
        return up to 5 questions from metadata['questions'] whose phase matches the input phase.
        """
        try:
            # phaseの文字列表現を取得
            phase_text = getattr(phase, "value", str(phase))
            print("phase_text", phase_text)

            # Embed the summary into a vector
            query_vec = self.embed_model.get_text_embedding(summary)
            # Search Qdrant for similar nodes (top 10)
            vs_query = VectorStoreQuery(
                query_embedding=query_vec,
                similarity_top_k=10,
            )

            results = self.vector_store.query(vs_query)

            matched_questions = []
            for node in results.nodes:
                questions_json = node.metadata.get("questions")
                if not questions_json:
                    continue
                for q in json.loads(questions_json):
                    q_phase_text = q.get("phase")
                    if str(q_phase_text) == phase_text:
                        matched_questions.append(q.get("question"))
                        if len(matched_questions) >= 5:
                            break
                if len(matched_questions) >= 5:
                    break
            return matched_questions
        except Exception as e:
            print("[ERROR] search_similar_questions:", e)
            traceback.print_exc()
            raise
