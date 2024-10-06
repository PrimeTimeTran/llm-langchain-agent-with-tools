import pandas as pd
from datasets import load_dataset
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
)

from setup.collection_utils import batch_insert
from setup.constants import EMBEDDINGS_LIMIT


def create_research_collection():
    # TODO: Use the vector_store to save documents to Milvus.
    # In this way we'll create our own embeddings
    # for the text as opposed to using those already
    # created which we don't know which embedding function was used
    name = "research_retriever"
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
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=EMBEDDINGS_LIMIT),
            FieldSchema("title", DataType.VARCHAR, max_length=500),
            FieldSchema("authors", DataType.VARCHAR, max_length=1500),
            FieldSchema("text", DataType.VARCHAR, max_length=5000),
            FieldSchema("abstract", DataType.VARCHAR, max_length=5000),
            FieldSchema("submitter", DataType.VARCHAR, max_length=100),
        ],
    )

    collection = Collection(
        auto_id=True,
        schema=schema,
        name=name,
        enable_dynamic_field=True,
    )

    print("Collection created successfully.")

    index_params = {
        "index_type": "AUTOINDEX",
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    collection.create_index(field_name="vector", index_params=index_params, timeout=None)

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
