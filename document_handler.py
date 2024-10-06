from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from setup.configs import cfg

docs = [
    Document(
        page_content="The best color is red",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="The best color is blue",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Loi is a programmer",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Cong is a programmer",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
]

db = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=cfg.get("OPENAI_API_KEY")))

query = "Who is a programmer?"
docs = db.similarity_search(query, k=2, filter={"genre": "science fiction"})

for doc in docs:
    print("doc", doc)
    print(doc.page_content)

query = "What is the best color?"
docs = db.similarity_search(query, k=2)

for doc in docs:
    print("doc", doc)
    print(doc.page_content)
