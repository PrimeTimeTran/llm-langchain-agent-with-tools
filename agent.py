import logging
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from langchain_openai import ChatOpenAI
from tools import Tools


AGENT_SYSTEM_TEMPLATE = """
You are a helpful agent that retrieves research papers and answers questions about them.
"""


def init_agent(
    llm: ChatOpenAI, tools: Tools
) -> CompiledGraph:
    agent_executor = create_react_agent(
        llm,
        tools,
        messages_modifier=SystemMessage(content=AGENT_SYSTEM_TEMPLATE),
    )

    logging.info("Agent initialized")
    return agent_executor

class AIMessage:
    def __init__(self, content: str, other_field: str):
        self.content = content
        self.other_field = other_field
        
def serialize_ai_message(message: AIMessage) -> dict:
    return {
        "content": message.content,
    }
