import time
from loguru import logger
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError
from config import config


class Embedder:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.EMBEDDING_MODEL
        self.dimensions = config.EMBEDDING_DIMENSIONS
        self.max_retries = config.MAX_RETRIES
        logger.info(
            f"✅ [EMBEDDER] Initialized — model={self.model}, dims={self.dimensions}"
        )

    def embed(self, text: str) -> list[float]:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.time()
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    dimensions=self.dimensions,
                )
                embedding = response.data[0].embedding
                elapsed = time.time() - start
                logger.info(
                    f"✅ [EMBEDDER] Query embedded ({len(embedding)} dims) in {elapsed:.2f}s"
                )
                return embedding

            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"⚠️  [EMBEDDER] Transient error on attempt {attempt}/{self.max_retries}: "
                    f"{type(e).__name__}. Retrying in {wait}s..."
                )
                time.sleep(wait)

            except Exception as e:
                logger.error(f"❌ [EMBEDDER] Non-retryable error: {e}")
                raise

        logger.error(f"❌ [EMBEDDER] All {self.max_retries} attempts failed")
        raise RuntimeError(
            f"Embedding failed after {self.max_retries} retries. Last error: {last_error}"
        )


embedder = Embedder()
