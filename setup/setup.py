from setup.agent import init_agent
from setup.configs import cfg
from setup.embedding import init_embeddings
from setup.llm import init_llm
from setup.tools import init_tools


def setup():
    embeddings = init_embeddings(cfg)
    tools, store = init_tools(cfg, embeddings)
    llm = init_llm(cfg, tools)
    agent = init_agent(llm, tools)
    return agent, llm, tools, store
