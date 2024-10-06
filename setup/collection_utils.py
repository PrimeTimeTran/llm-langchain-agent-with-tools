from pymilvus import (
    utility,
    Collection,
)

from setup.constants import EMBEDDINGS_LIMIT


def batch_insert(collection, dataset_df):
    batch_size = 100
    insert_data = []
    for _, row in dataset_df.iterrows():
        embedding = row["embedding"]
        if len(embedding) > EMBEDDINGS_LIMIT:
            embedding = embedding[:EMBEDDINGS_LIMIT]
        elif len(embedding) < EMBEDDINGS_LIMIT:
            embedding = embedding + [0.0] * (EMBEDDINGS_LIMIT - len(embedding))
        insert_data.append(
            {
                "vector": embedding,
                "title": row["title"],
                "authors": row["authors"],
                "abstract": row["abstract"],
                "text": row["abstract"],
                "submitter": row["submitter"],
            }
        )
        if len(insert_data) == batch_size:
            try:
                collection.insert(insert_data)
                print(f"Inserted {len(insert_data)} records successfully.")
            except Exception as e:
                print(f"Failed to insert batch: {e}")
            insert_data = []

    if insert_data:
        try:
            collection.insert(insert_data)
            print(f"Inserted {len(insert_data)} records successfully.")
        except Exception as e:
            print(f"Failed to insert last batch: {e}")


def clean_collections(collections_to_drop):
    for collect_name in collections_to_drop:
        if collect_name in utility.list_collections():
            try:
                Collection(collect_name).drop()
                print(f"Existing collection '{collect_name}' dropped successfully.")
            except Exception as e:
                print(f"Failed to drop existing collection: {e}")
