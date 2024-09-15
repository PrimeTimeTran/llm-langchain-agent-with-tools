from langchain.agents import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools.simple import Tool

from langchain_community.document_loaders import ArxivLoader
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch


from database import DB_NAME, MONGO_URI, COLLECTION_NAME, ATLAS_VECTOR_SEARCH_INDEX_NAME


@tool
def get_metadata_information_from_arxiv(word: str) -> list:
    """
    Fetches and returns metadata for a maximum of ten documents from arXiv matching the given query word.

    Args:
    word (str): The search query to find relevant documents on arXiv.

    Returns:
    list: Metadata about the documents matching the query.
    """
    docs = ArxivLoader(query=word, load_max_docs=10).load()
    metadata_list = [doc.metadata for doc in docs]
    return metadata_list


@tool
def get_information_from_arxiv(word: str) -> list:
    """
    Fetches and returns metadata for a single research paper from arXiv matching the given query word, which is the ID of the paper, for example: 704.0001.

    Args:
    word (str): The search query to find the relevant paper on arXiv using the ID.

    Returns:
    list: Data about the paper matching the query.
    """
    doc = ArxivLoader(query=word, load_max_docs=1).load()
    return doc


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    text_key="abstract",
    embedding=embedding_model,
    connection_string=MONGO_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="knowledge_base",
    description="This serves as the base knowledge source of the agent and contains some records of research papers from Arxiv. This tool is used as the first step for exploration and research efforts.",
)


# tools = [
#     get_metadata_information_from_arxiv,
#     get_information_from_arxiv,
#     retriever_tool,
# ]


tools = [
  Tool(
    func=get_metadata_information_from_arxiv,
    name="get_metadata_information_from_arxiv",
    description="get_metadata_information_from_arxiv"
  ),
  Tool(
    func=get_information_from_arxiv,
    name="get_information_from_arxiv",
    description="get_information_from_arxiv"
  ),
  Tool(
    func=retriever_tool,
    name="retriever_tool",
    description="retriever_tool"
  ),
]
