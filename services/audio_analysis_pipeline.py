from __future__ import annotations
import os
from typing import List
from openai import OpenAI
import boto3
from models import (
    Transcript,
    QAResult,
    ProductInfo,
    ProcessedAudioData,
)
from managers.audio_manager import AudioManager
from helpers.vectordb import VectorStoreHandler
from helpers.question_extractor import QuestionExtractor
from managers.summary_generator import SummaryGenerator
from helpers.product_info_extractor import ProductInfoExtractor
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class AudioProcessor:
    """End‑to‑end pipeline for processing call recordings stored in S3."""

    def __init__(
        self,
        *,
        openai_client: OpenAI,
        s3_client: "boto3.client",
        s3_bucket: str,
        s3_prefix: str,
        db_url: str | None = None,
    ) -> None:
        self.openai = openai_client
        self.s3 = s3_client
        self.bucket = s3_bucket
        self.prefix = s3_prefix

        self.audio_manager = AudioManager(s3_client=self.s3, s3_bucket=self.bucket, openai_client=self.openai)
        self.question_extractor = QuestionExtractor(openai_client=self.openai)
        self.product_info_extractor = ProductInfoExtractor(
            openai_client=self.openai,
            db_url=db_url or os.getenv("DATABASE_URL")
        )
        self.summary_generator = SummaryGenerator(openai_client=self.openai)
        self.vector_store_handler = VectorStoreHandler()

    def run(self) -> None:
        """Execute the full pipeline over all wav files under the prefix, in parallel."""
        print("[AudioProcessor]  Pipeline start")
        keys = self._list_audio_files()
        processing_count = 0
        processing_count_lock = threading.Lock()
        completed_count = 0
        completed_count_lock = threading.Lock()
        start_count = 0
        start_count_lock = threading.Lock()
        total = len(keys)
        print(f"[AudioProcessor] {total} files to process.")
        def wrapped_process_one_file(key):
            nonlocal processing_count, start_count
            with processing_count_lock:
                processing_count += 1
            with start_count_lock:
                start_count += 1
                print(f"[AudioProcessor] Start processing {start_count}/{total} files... (Currently running: {processing_count})")
            try:
                self._process_one_file(key)
            finally:
                with processing_count_lock:
                    processing_count -= 1
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(wrapped_process_one_file, key) for key in keys]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[AudioProcessor] Error: {e}")
                with completed_count_lock:
                    completed_count += 1
                    print(f"[AudioProcessor] Progress: {completed_count}/{total} files completed.")
        print("[AudioProcessor]  Pipeline finished")

    def _process_one_file(self, key: str) -> None:
        if self.vector_store_handler.exists_by_filename(key):
            print(f"[AudioProcessor] Skipping {key} (already exists in Qdrant)")
            return
        local_path = self.audio_manager.preprocess(key)
        transcript: Transcript = self.audio_manager.transcribe(local_path)
        questions: List[QAResult] = self.question_extractor.extract(transcript.text)
        product_info: List[ProductInfo] = self.product_info_extractor.extract(transcript.text)
        summary: str = self.summary_generator.generate(transcript.text)
        processed = ProcessedAudioData(
            file_name=key,
            questions=questions,
            product_info=product_info,
            summary=summary,
        )
        self.vector_store_handler.store_from_processed_data(processed)

    def _list_audio_files(self) -> List[str]:
        paginator = self.s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
        keys = []
        for page in page_iterator:
            keys.extend([
                obj["Key"]
                for obj in page.get("Contents", [])
                if obj["Key"].lower().endswith(".wav")
            ])
        return keys
