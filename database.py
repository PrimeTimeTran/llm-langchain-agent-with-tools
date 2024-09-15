import os
from pymongo import MongoClient, TEXT
from pymongo.operations import SearchIndexModel
from datetime import datetime


MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)

DB_NAME = "agent_demo"
COLLECTION_NAME = "knowledge"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client.get_database(DB_NAME).get_collection(COLLECTION_NAME)

search_index_model = SearchIndexModel(
  definition={
    "fields": [
      {
        "type": "vector",
        "numDimensions": 256,
        "path": "embedding",
        "similarity": "euclidean | cosine | dotProduct"
      }
    ]
  },
  name="vector_index",
  type="vectorSearch",
)

index_model = [("field_name", TEXT)]
result = collection.create_index(index_model)

def save_message(session_id: str, message_type: str, content: str):
    document = {
        "SessionId": session_id,
        "timestamp": datetime.utcnow(),
        "message_type": message_type,
        "content": content
    }
    collection.insert_one(document)
