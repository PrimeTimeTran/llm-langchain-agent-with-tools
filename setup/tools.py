from langchain.tools.retriever import create_retriever_tool
from langchain_core.embeddings import Embeddings
from langchain_core.tools.simple import Tool

from setup.vector_store import init_vector_store

type Tools = list[Tool]


def init_tools(cfg: dict, embeddings: Embeddings):
    tools, store = init_research_paper_retriever(cfg, embeddings)
    return [tools], store


def init_research_paper_retriever(cfg: dict, embeddings: Embeddings):
    name = "research_retriever"
    vector_store = init_vector_store(cfg, name, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    tool = create_retriever_tool(
        retriever,
        name,
        """Searches the dataset and returns research papers matching query term.""",
    )

    return tool, vector_store
