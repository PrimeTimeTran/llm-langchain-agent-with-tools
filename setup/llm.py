from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs.llm_result import LLMResult
from langchain_openai import ChatOpenAI

from setup.tools import Tools


def init_llm(cfg: dict, tools: Tools) -> ChatOpenAI:
    llm = init_openai_chat(cfg, tools)
    return llm


def init_openai_chat(cfg: dict, tools: Tools) -> ChatOpenAI:
    cb = ChatCallbackHandler()
    model = ChatOpenAI(
        model=cfg.get("OPENAI_GPT_MODEL", "gpt-3.5-turbo"),
        api_key=cfg.get("OPENAI_API_KEY"),
        max_tokens=cfg.get("OPENAI_MAX_TOKENS", 2048),
        callbacks=[cb],
    ).bind_tools(tools)

    return model


class ChatCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        if inputs is None:
            print("Warning: 'inputs' is None in on_chain_start")
        else:
            pass
            # print("Inputs:", inputs)
