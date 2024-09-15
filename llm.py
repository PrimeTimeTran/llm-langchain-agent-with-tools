import os
from typing import Any, Dict, List
from langchain_core.outputs.llm_result import LLMResult
from langchain_fireworks.chat_models import ChatFireworks
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# llm = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     api_key=OPENAI_API_KEY,
#     max_tokens=2048,
# )

llm = ChatFireworks(
    max_tokens=256,
    api_key=FIREWORKS_API_KEY, 
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
)


class ChatCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        if inputs is None:
            print("Warning: 'inputs' is None in on_chain_start")
        else:
            pass
            # print("Inputs:", inputs)
