import os
import pandas as pd
from datasets import load_dataset
from database import collection

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

data = load_dataset("MongoDB/subset_arxiv_papers_with_embeddings")
dataset_df = pd.DataFrame(data["train"])

records = dataset_df.to_dict('records')
collection.insert_many(records)
print("Data ingestion into MongoDB completed")

