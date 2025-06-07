import os
import pytest
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall
from utils import get_llm_response, load_json_test_data

@pytest.fixture()
def getdata(request):
    test_data=request.param
    responseDict = get_llm_response(test_data)
    sample = SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=[
            responseDict["retrieved_docs"][0]["page_content"],
            responseDict["retrieved_docs"][1]["page_content"],
            responseDict["retrieved_docs"][2]["page_content"],
            responseDict["retrieved_docs"][3]["page_content"]
        ],
        reference=test_data["reference"]
    )
    return sample

@pytest.mark.asyncio
@pytest.mark.parametrize("getdata",
                         load_json_test_data(),
                         indirect = True
                         )

async def test_context_recall(llm_wrapper, getdata):
    context_recall = LLMContextRecall(llm=llm_wrapper)
    score = await context_recall.single_turn_ascore(getdata)
    print(score)
    assert score > 0.5