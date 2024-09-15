import os
from pymongo import MongoClient


MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)

DB_NAME = "agent_demo"
COLLECTION_NAME = "knowledge"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client.get_database(DB_NAME).get_collection(COLLECTION_NAME)
