import os

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ragas.llms import LangchainLLMWrapper

@pytest.fixture()
def llm_wrapper():
    os.environ["OPENAI_API_KEY"] = "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"
    api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     temperature=0,
                     api_key=SecretStr(api_key)
                     )
    langchain_llm = LangchainLLMWrapper(llm)
    return langchain_llm