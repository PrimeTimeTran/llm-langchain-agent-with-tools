import logging
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from langchain_openai import ChatOpenAI
from setup.tools import Tools

AGENT_SYSTEM_TEMPLATE = """
You are a helpful agent that retrieves research papers and answers questions about them.
"""


def init_agent(llm: ChatOpenAI, tools: Tools) -> CompiledGraph:
    agent = create_react_agent(
        model=llm,
        tools=tools,
        messages_modifier=SystemMessage(content=AGENT_SYSTEM_TEMPLATE),
    )

    logging.info("Agent initialized")
    return agent
