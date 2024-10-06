import logging
from pymilvus import connections
from langchain_milvus import Milvus
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from setup.configs import cfg
from setup.embedding import init_embeddings
from setup.collection_utils import clean_collections
from setup.create_wikipedia_collection import create_wikipedia_collection
from setup.create_research_collection import create_research_collection


def init_milvus(embeddings: Embeddings) -> Milvus:
    db = Milvus(
        auto_id=True,
        embedding_function=embeddings,
        collection_name="vector",
        connection_args={
            "uri": cfg.get("ZILLIZ_CLOUD_URI"),
            "token": cfg.get("ZILLIZ_CLOUD_API_KEY"),
        },
    )
    print("Connected to Milvus server successfully.")
    return db


def connect_to_db(embeddings):
    try:
        init_milvus(embeddings)
    except Exception as e:
        print(f"Failed to connect to Milvus server: {e}")


def init_collections(embeddings: Embeddings):
    try:
        # Why does running not reflect this function call?
        connect_to_db(embeddings)
        # collections_to_drop = []
        # clean_collections(collections_to_drop)
        # create_wikipedia_collection()
        # create_research_collection()
    except Exception as e:
        print(f"Failed to create collection or index: {e}")
        return


embeddings = init_embeddings(cfg)
init_collections(embeddings)
