from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain.prompts import PromptTemplate

from setup.setup import setup

agent, llm, tools, store = setup()

system_msg = SystemMessage(
    content="""
    You are an assistant that reviews research papers for key terms.
    If a paper does not include the keywords you do not retrieve or return it in your response.
    You include the sentence which had that keyword in it from the paper in your response.
    """
)
user_query = "Give me info on the 'string theory'. Summarize their text in the response"
prompt = PromptTemplate(
    input_variables=["keyword"], template="Find any paper with the keyword {keyword}."
)

msgs = [
    system_msg,
    HumanMessage(content=user_query),
]

configurable = {
    "query": user_query,
    "messages": msgs,
    "recursion_limit": 25,
    "search_kwargs": {"k": 5},
}
config = RunnableConfig(configurable)

response = agent.invoke(config)

if response.get("messages"):
    print("length", len(response.get("messages")))
    messages = []
    for doc in response["messages"]:
        tool_calls = getattr(response, "tool_calls", None)
        if doc.content:
            message = AIMessage(content=doc.content, id=doc.id)
            messages.append(message)
        else:
            pass
            # tool_output = tools[0].invoke(doc.content)
            # messages.append(ToolMessage(tool_output, tool_call_id=doc.id))
    for msg in messages:
        print("msg", msg.content)
else:
    print("No messages found.")
