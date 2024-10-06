import logging

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from pymilvus import connections

from setup.collection_utils import clean_collections
from setup.configs import cfg
from setup.create_research_collection import create_research_collection
from setup.create_wikipedia_collection import create_wikipedia_collection
from setup.embedding import init_embeddings


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


def connect_to_db(embeddings) -> VectorStore:
    try:
        return 1
        return init_milvus(embeddings)
    except Exception as e:
        print(f"Failed to connect to Milvus server: {e}")


def init_collections(embeddings: Embeddings):
    try:
        vector_store: VectorStore = connect_to_db(embeddings)
        # collections_to_drop = []
        # clean_collections(collections_to_drop)
        create_wikipedia_collection(vector_store)
        # create_research_collection()
    except Exception as e:
        print(f"Failed to create collection or index: {e}")
        return


embeddings = init_embeddings(cfg)
init_collections(embeddings)
