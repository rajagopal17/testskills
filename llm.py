import time
from loguru import logger
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError
from config import config

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question using ONLY the provided context chunks.\n"
    "- Cite sources as [Chunk 1], [Chunk 2], etc.\n"
    "- If the context does not contain enough information, say: "
    "\"I don't have enough context to answer this.\"\n"
    "- Be concise and factual."
)


class LLMSynthesizer:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS
        self.system_prompt = config.SYSTEM_PROMPT or DEFAULT_SYSTEM_PROMPT
        self.max_retries = config.MAX_RETRIES
        logger.info(
            f"✅ [LLM] Initialized — model={self.model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            score = chunk.get("rerank_score", chunk.get("score", 0.0))
            parts.append(f"[Chunk {i}] (score: {score:.4f})\n{chunk['chunk_text']}")
        return "\n\n---\n\n".join(parts)

    def synthesize(self, query: str, chunks: list[dict]) -> str | None:
        if not chunks:
            logger.warning("⚠️  [LLM] No chunks to synthesize from — skipping LLM call")
            return None

        context = self._build_context(chunks)
        user_message = f"Context:\n\n{context}\n\nQuestion: {query}"

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                answer = response.choices[0].message.content
                tokens = response.usage.total_tokens
                elapsed = time.time() - start
                logger.info(
                    f"✅ [LLM] Answer synthesized ({tokens} tokens, "
                    f"model={self.model}) in {elapsed:.2f}s"
                )
                return answer

            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"⚠️  [LLM] Transient error on attempt {attempt}/{self.max_retries}: "
                    f"{type(e).__name__}. Retrying in {wait}s..."
                )
                time.sleep(wait)

            except Exception as e:
                logger.error(f"❌ [LLM] Non-retryable error: {e}")
                return None

        logger.error(
            f"❌ [LLM] All {self.max_retries} attempts failed — returning chunks only. "
            f"Last error: {last_error}"
        )
        return None


llm_synthesizer = LLMSynthesizer()
