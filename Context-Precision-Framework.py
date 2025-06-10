import os
import pytest
import asyncio
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference
from utils import get_llm_response


@pytest.fixture()
def getdata(request):
    #question = "How many articles are there in the Selenium Webdriver python course?"
    test_data = request.param
    responseDict = get_llm_response(test_data)
    # responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
    #                                 json={
    #                                     "question": test_data["question"],
    #                                     "chat_history": [
    #                                     ]
    #                                 }
    #                                 ).json()
    # print(responseDict)
    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[
            responseDict["retrieved_docs"][0]["page_content"],
            responseDict["retrieved_docs"][1]["page_content"],
            responseDict["retrieved_docs"][2]["page_content"]
        ]
    )
    return sample


@pytest.mark.asyncio
@pytest.mark.parametrize("getdata",
                         [
                             {
                                 "question": "How many articles are there in the Selenium Webdriver python course?"
                             }
                         ], indirect=True
                         )
async def test_context_precision(llm_wrapper, getdata):
    context_precision = LLMContextPrecisionWithoutReference(llm=llm_wrapper)
    score = await context_precision.single_turn_ascore(getdata)
    print(score)
    # Assertion
    assert score > 0.8
