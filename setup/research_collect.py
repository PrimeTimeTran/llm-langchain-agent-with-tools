import pandas as pd
from datasets import load_dataset
from pymilvus import (
    utility,
    DataType,
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
)

from configs import cfg
from utils import batch_insert
from embedding import init_embeddings


def init_database_collection(embeddings):
    try:
        connections.connect(
            alias="default",
            uri=cfg.get("ZILLIZ_CLOUD_URI"),
            token=cfg.get("ZILLIZ_CLOUD_API_KEY"),
        )
        print("Connected to Milvus server successfully.")
    except Exception as e:
        print(f"Failed to connect to Milvus server: {e}")

    collection_name = "research_retriever"
    if collection_name in utility.list_collections():
        try:
            Collection(collection_name).drop()
            print(f"Existing collection '{collection_name}' dropped successfully.")
        except Exception as e:
            print(f"Failed to drop existing collection: {e}")
    try:
        id_field = FieldSchema(
            name="id",
            auto_id=True,
            is_primary=True,
            dtype=DataType.INT64,
            description="primary id",
        )

        schema = CollectionSchema(
            [
                id_field,
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=256),
                FieldSchema("title", DataType.VARCHAR, max_length=500),
                FieldSchema("authors", DataType.VARCHAR, max_length=1500),
                FieldSchema("abstract", DataType.VARCHAR, max_length=5000),
                FieldSchema("submitter", DataType.VARCHAR, max_length=100),
            ],
        )

        collection = Collection(name=collection_name, schema=schema, auto_id=True)
        print("Collection created successfully.")

        index_params = {
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }

        collection.create_index(
            field_name="vector", index_params=index_params, timeout=None
        )
        print("Index created successfully.")
    except Exception as e:
        print(f"Failed to create collection or index: {e}")
        return

    try:
        data = load_dataset("MongoDB/subset_arxiv_papers_with_embeddings")
        dataset_df = pd.DataFrame(data["train"])
        for col in dataset_df.columns.tolist():
            print("col", col)
        batch_insert(collection, dataset_df)
        print("Data inserted successfully.")
        collection.load()
        print("Collection loaded successfully.")
    except Exception as e:
        print(f"Failed to load collection: {e}")


embeddings = init_embeddings(cfg)
init_database_collection(embeddings)
