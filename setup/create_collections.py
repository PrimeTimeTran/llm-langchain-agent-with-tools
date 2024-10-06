from pymilvus import connections
from langchain_core.embeddings import Embeddings

from configs import cfg
from embedding import init_embeddings
from collection_utils import clean_collections
from create_wikipedia_collection import create_wikipedia_collection
from create_research_collection import create_research_collection


def connect_to_milvus():
    try:
        uri = cfg.get("ZILLIZ_CLOUD_URI")
        token = cfg.get("ZILLIZ_CLOUD_API_KEY")
        connections.connect(
            alias="default",
            uri=uri,
            token=token,
        )
        print("Connected to Milvus server successfully.")
    except Exception as e:
        print(f"Failed to connect to Milvus server: {e}")


def init_collections(embeddings: Embeddings):
    try:
        # connect_to_milvus()
        # collections_to_drop = []
        # clean_collections(collections_to_drop)
        create_wikipedia_collection()
        # create_research_collection()
    except Exception as e:
        print(f"Failed to create collection or index: {e}")
        return


embeddings = init_embeddings(cfg)
init_collections(embeddings)
