import os
import sys
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self):
        # PostgreSQL
        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = int(os.getenv("DB_PORT", "5432"))
        self.DB_NAME = os.getenv("DB_NAME", "")
        self.DB_USER = os.getenv("DB_USER", "")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "")
        self.DB_SCHEMA = os.getenv("DB_SCHEMA", "public")

        # Table & Columns
        self.TABLE_NAME = os.getenv("TABLE_NAME", "")
        self.TEXT_COLUMN = os.getenv("TEXT_COLUMN", "chunk_text")
        self.EMBEDDING_COLUMN = os.getenv("EMBEDDING_COLUMN", "embedding")
        self.METADATA_COLUMNS = [
            c.strip()
            for c in os.getenv("METADATA_COLUMNS", "").split(",")
            if c.strip()
        ]

        # pgvector / Index
        self.PGVECTOR_INDEX_TYPE = os.getenv("PGVECTOR_INDEX_TYPE", "hnsw").lower()
        self.PGVECTOR_DISTANCE_METRIC = os.getenv("PGVECTOR_DISTANCE_METRIC", "cosine").lower()
        self.PGVECTOR_INDEX_NAME = os.getenv("PGVECTOR_INDEX_NAME", "")
        self.HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))
        self.HNSW_M = int(os.getenv("HNSW_M", "16"))
        self.IVFFLAT_NPROBE = int(os.getenv("IVFFLAT_NPROBE", "10"))
        self.IVFFLAT_LISTS = int(os.getenv("IVFFLAT_LISTS", "100"))

        # Search
        self.DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

        # Embeddings
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))

        # LLM
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")

        # Reranker
        self.RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "5"))

        # Retry
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

        # FastAPI Debug Server
        self.DEBUG_API_HOST = os.getenv("DEBUG_API_HOST", "127.0.0.1")
        self.DEBUG_API_PORT = int(os.getenv("DEBUG_API_PORT", "8000"))

    def validate(self):
        required = {
            "DB_NAME": self.DB_NAME,
            "DB_USER": self.DB_USER,
            "DB_PASSWORD": self.DB_PASSWORD,
            "TABLE_NAME": self.TABLE_NAME,
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required .env variables: {', '.join(missing)}")

        valid_index_types = {"hnsw", "ivfflat", "flat"}
        if self.PGVECTOR_INDEX_TYPE not in valid_index_types:
            raise ValueError(
                f"PGVECTOR_INDEX_TYPE must be one of {valid_index_types}, "
                f"got: '{self.PGVECTOR_INDEX_TYPE}'"
            )

        valid_metrics = {"cosine", "l2", "inner_product"}
        if self.PGVECTOR_DISTANCE_METRIC not in valid_metrics:
            raise ValueError(
                f"PGVECTOR_DISTANCE_METRIC must be one of {valid_metrics}, "
                f"got: '{self.PGVECTOR_DISTANCE_METRIC}'"
            )

        logger.info(
            f"✅ [CONFIG] Validated — db={self.DB_NAME}, table={self.TABLE_NAME}, "
            f"index={self.PGVECTOR_INDEX_TYPE}, metric={self.PGVECTOR_DISTANCE_METRIC}"
        )


config = Config()
