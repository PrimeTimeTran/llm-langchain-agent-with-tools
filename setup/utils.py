def batch_insert(collection, dataset_df):
    batch_size = 100
    insert_data = []
    for idx, row in dataset_df.iterrows():
        insert_data.append(
            {
                "vector": [float(val) for val in row["embedding"]],
                "title": row["title"],
                "authors": row["authors"],
                "abstract": row["abstract"],
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
