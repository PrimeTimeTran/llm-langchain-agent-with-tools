from langchain_core.messages import HumanMessage, AIMessage
from database import collection

def print_history(response):
    chat_history = response.get("chat_history", [])

    if len(chat_history) > 0:
        last_message = chat_history[-1]
        print(
            "Last message content:",
            last_message.content if hasattr(last_message, "content") else "No content",
        )
    else:
        print("Not enough messages in chat history to access -1 index")

    if len(chat_history) > 1:
        second_to_last_message = chat_history[-2]
        print(
            "Second-to-last message content:",
            second_to_last_message.content
            if hasattr(second_to_last_message, "content")
            else "No content",
        )
    else:
        print("Not enough messages in chat history to access -2 index")

def format_messages(documents):
    messages = []
    for doc in documents:
        if doc['message_type'] == 'user':
            messages.append(HumanMessage(content=doc['content']))
        elif doc['message_type'] == 'ai':
            messages.append(AIMessage(content=doc['content']))
    return messages


async def get_formatted_history(session_id: str):
    docs = collection.find({"SessionId": session_id})
    return format_messages(docs)
