import time
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from loguru import logger
from config import config

# Maps distance metric name → pgvector SQL operator
DISTANCE_OPS = {
    "cosine": "<=>",
    "l2": "<->",
    "inner_product": "<#>",
}


def _score_from_distance(distance: float, metric: str) -> float:
    """Convert raw pgvector distance to a 0–1 similarity score."""
    if metric == "cosine":
        # cosine distance ∈ [0, 2]; similarity = 1 - distance
        return round(1.0 - float(distance), 6)
    elif metric == "l2":
        # normalise to (0, 1]
        return round(1.0 / (1.0 + float(distance)), 6)
    elif metric == "inner_product":
        # pgvector returns negative inner product
        return round(-float(distance), 6)
    return round(float(distance), 6)


class Searcher:
    def __init__(self):
        self.max_retries = config.MAX_RETRIES
        logger.info(
            f"✅ [SEARCHER] Initialized — index={config.PGVECTOR_INDEX_TYPE}, "
            f"metric={config.PGVECTOR_DISTANCE_METRIC}, "
            f"threshold={config.SIMILARITY_THRESHOLD}"
        )

    def _connect(self):
        return psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )

    def _apply_index_params(self, cursor):
        """Set session-level index parameters for the configured index type."""
        index_type = config.PGVECTOR_INDEX_TYPE
        if index_type == "hnsw":
            cursor.execute(f"SET hnsw.ef_search = {config.HNSW_EF_SEARCH};")
            logger.debug(f"[SEARCHER] SET hnsw.ef_search = {config.HNSW_EF_SEARCH}")
        elif index_type == "ivfflat":
            cursor.execute(f"SET ivfflat.probes = {config.IVFFLAT_NPROBE};")
            logger.debug(f"[SEARCHER] SET ivfflat.probes = {config.IVFFLAT_NPROBE}")
        elif index_type == "flat":
            # Force sequential scan — guarantees exact results regardless of index
            cursor.execute("SET enable_indexscan = off;")
            logger.debug("[SEARCHER] SET enable_indexscan = off (flat/exact scan)")

    def search(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict]:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._execute_search(embedding, top_k, filters)
            except psycopg2.OperationalError as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"⚠️  [SEARCHER] DB connection error on attempt {attempt}/{self.max_retries}: "
                    f"{e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            except Exception as e:
                logger.error(f"❌ [SEARCHER] Non-retryable error: {e}")
                raise

        logger.error(f"❌ [SEARCHER] All {self.max_retries} DB attempts failed")
        raise RuntimeError(
            f"Search failed after {self.max_retries} retries. Last error: {last_error}"
        )

    def _execute_search(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict]:
        start = time.time()
        metric = config.PGVECTOR_DISTANCE_METRIC
        op = DISTANCE_OPS.get(metric, "<=>")
        schema = config.DB_SCHEMA
        table = config.TABLE_NAME
        text_col = config.TEXT_COLUMN
        emb_col = config.EMBEDDING_COLUMN
        meta_cols = config.METADATA_COLUMNS

        select_cols = ["id", text_col] + meta_cols
        select_sql = ", ".join(select_cols)

        # Build parameterised WHERE clause from metadata filters
        where_clauses: list[str] = []
        where_values: list = []
        if filters:
            for col, val in filters.items():
                if col in meta_cols:
                    where_clauses.append(f"{col} = %s")
                    where_values.append(val)
                else:
                    logger.warning(
                        f"⚠️  [SEARCHER] Filter column '{col}' not in METADATA_COLUMNS — skipped"
                    )

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        query_vec = np.array(embedding, dtype=np.float32)

        sql = f"""
            SELECT {select_sql},
                   {emb_col} {op} %s AS _distance
            FROM {schema}.{table}
            {where_sql}
            ORDER BY {emb_col} {op} %s ASC
            LIMIT %s
        """
        params = where_values + [query_vec, query_vec, top_k]

        conn = self._connect()
        try:
            register_vector(conn)
            with conn.cursor() as cur:
                self._apply_index_params(cur)
                cur.execute(sql, params)
                rows = cur.fetchall()
                col_names = [desc[0] for desc in cur.description]
        finally:
            conn.close()

        results = []
        for row in rows:
            row_dict = dict(zip(col_names, row))
            distance = row_dict.pop("_distance")
            score = _score_from_distance(distance, metric)
            if score >= config.SIMILARITY_THRESHOLD:
                results.append(
                    {
                        "id": str(row_dict.get("id", "")),
                        "chunk_text": row_dict.get(text_col, ""),
                        "score": score,
                        "metadata": {k: row_dict[k] for k in meta_cols if k in row_dict},
                    }
                )

        elapsed = time.time() - start
        logger.info(
            f"✅ [SEARCHER] Retrieved {len(results)} chunks above threshold "
            f"(top_k={top_k}, threshold={config.SIMILARITY_THRESHOLD}, "
            f"index={config.PGVECTOR_INDEX_TYPE}) in {elapsed:.2f}s"
        )
        return results


searcher = Searcher()
