from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful research assistant"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
  ]
)
