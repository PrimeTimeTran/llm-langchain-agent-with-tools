from configs.configs import cfg

from llm import init_llm
from agent import init_agent
from tools import init_tools
from embedding import init_embeddings

def setup():
    print('setup')
    llm = init_llm(cfg)
    embeddings = init_embeddings(cfg)
    tools = init_tools(cfg, embeddings)
    agent = init_agent(llm, tools)
    print('Agent Initialized')
    return agent

setup()
