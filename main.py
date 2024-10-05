from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain.prompts import PromptTemplate

from setup.setup import setup

agent, llm, tools, store = setup()

system_msg = SystemMessage(
    content="You are a helpful assistant that retrieves research papers."
)
user_query = (
    "Give me the first 3 papers on climate change. Summarize their text in the response"
)
prompt = PromptTemplate(
    input_variables=["keyword"], template="Find any paper with the keyword {keyword}."
)

msgs = [
    system_msg,
    HumanMessage(content=user_query),
]

query = "Climate change"
configurable = {
    "query": query,
    "messages": msgs,
    "recursion_limit": 25,
    "search_kwargs": {"k": 5},
}
config = RunnableConfig(configurable)

response = agent.invoke(config)

print("response", response)

if response.get("messages"):
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
