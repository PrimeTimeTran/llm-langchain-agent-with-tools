
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory


from database import MONGO_URI, DB_NAME, client

db = client[DB_NAME]
collection = db["history"]

def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
  return MongoDBChatMessageHistory(MONGO_URI, session_id, database_name=DB_NAME, collection_name="history")



memory = ConversationBufferMemory(
  memory_key="chat_history",
  chat_memory=get_session_history("my-session"),
  return_messages=True
)

config = {"configurable": {"thread_id": "my-session"}}


