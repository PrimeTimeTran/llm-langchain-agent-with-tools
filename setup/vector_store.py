import logging

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus


def init_vector_store(cfg: dict, kind: str, embeddings: Embeddings) -> VectorStore:
    return init_mv_vs(cfg, kind, embeddings)


def init_mv_vs(cfg: dict, kind: str, embeddings: Embeddings) -> Milvus:
    mv = Milvus(
        auto_id=True,
        embedding_function=embeddings,
        collection_name=kind or "vector",
        connection_args={
            "uri": cfg.get("ZILLIZ_CLOUD_URI"),
            "token": cfg.get("ZILLIZ_CLOUD_API_KEY"),
        },
    )

    logging.info(f"Milvus {kind} initialized")
    return mv
