import time
from loguru import logger
from config import config


class Reranker:
    def __init__(self):
        self.model = None
        self.top_n = config.RERANKER_TOP_N
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"[RERANKER] Loading cross-encoder: {config.RERANKER_MODEL}")
            self.model = CrossEncoder(config.RERANKER_MODEL)
            logger.info(f"✅ [RERANKER] Model loaded: {config.RERANKER_MODEL}")
        except Exception as e:
            logger.error(
                f"❌ [RERANKER] Failed to load model '{config.RERANKER_MODEL}': {e}. "
                f"Reranking will be skipped."
            )
            self.model = None

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            logger.warning("⚠️  [RERANKER] No chunks provided — nothing to rerank")
            return []

        if not self.model:
            logger.warning(
                "⚠️  [RERANKER] Model unavailable — returning top-N un-reranked chunks"
            )
            return chunks[: self.top_n]

        try:
            start = time.time()
            pairs = [(query, chunk["chunk_text"]) for chunk in chunks]
            scores = self.model.predict(pairs)

            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)

            reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            top_chunks = reranked[: self.top_n]

            elapsed = time.time() - start
            top_score = top_chunks[0]["rerank_score"] if top_chunks else 0.0
            logger.info(
                f"✅ [RERANKER] Reranked {len(chunks)} → top {len(top_chunks)} chunks "
                f"(best rerank score: {top_score:.4f}) in {elapsed:.2f}s"
            )
            return top_chunks

        except Exception as e:
            logger.error(f"❌ [RERANKER] Reranking failed: {e}")
            logger.warning("⚠️  [RERANKER] Falling back to un-reranked results")
            return chunks[: self.top_n]


reranker = Reranker()
