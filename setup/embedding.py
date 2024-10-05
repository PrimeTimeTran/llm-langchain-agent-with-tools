import logging
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def init_embeddings(cfg: dict) -> Embeddings:
    embeddings = init_openai_embeddings(cfg)
    return embeddings


def init_openai_embeddings(cfg: dict) -> OpenAIEmbeddings:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=cfg.get("OPENAI_API_KEY"),
        dimensions=3072,
    )

    logging.info("OpenAI Embeddings model initialized")
    return embeddings
