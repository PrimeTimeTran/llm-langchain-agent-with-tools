import asyncio
from langchain_core.messages import HumanMessage,AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent

from llm import llm
from tools import tools
from prompts import prompt
from memory import memory
from utils import get_formatted_history

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


async def simulate_agent_calls():
    chat_history = await get_formatted_history('my-session')

    initial_input = "Retrieve list of research papers on the Dogs"
    response = agent_executor.invoke(
        {
            "input": initial_input,
            "chat_history": chat_history,
        },
    )
    memory.chat_memory.add_user_message(HumanMessage(content=initial_input))

    if isinstance(response.get("chat_history"), list):
        response_content = response.get("output", "")
    else:
        response_content = "No response from agent"

    memory.save_context({"input": initial_input}, {"output": response_content})
    memory.chat_memory.add_ai_message(AIMessage(content=response_content))

    updated_history = await memory.chat_memory.aget_messages()

    subsequent_input = "Get me the abstract of the first paper on the list from my previous request/prompt"
    subsequent_response = agent_executor.invoke(
        {
            "input": subsequent_input,
            "chat_history": updated_history,
        },
    )
    print("Subsequent response Length:", len(subsequent_response))
    print("Subsequent response:", subsequent_response)


asyncio.run(simulate_agent_calls())
