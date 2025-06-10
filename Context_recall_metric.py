import os
import pytest
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall


@pytest.mark.asyncio
async def test_context_recall():  # function test case

    # connection to openAI GPT
    # Load your .env file where OPENAI_API_KEY is stored
    #load_dotenv()
    #client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    os.environ[
        "OPENAI_API_KEY"] = "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"

    # Set up the base LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     temperature=0
                     #openai_api_key=os.getenv("OPENAI_API_KEY")
                     )
    # Wrap it for use in Ragas
    langchain_llm = LangchainLLMWrapper(llm)
    # Initialize the metric
    context_recall = LLMContextRecall(llm=langchain_llm)  # class LLMContextRecall  object =context_recall

    # 2. Feed data
    question = "How many articles are there in the Selenium Webdriver python course?"
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": question,
                                     "chat_history": [
                                     ]
                                 }
                                 ).json()
    print(responseDict)

    #reference which is ground truth , for testing we have expedted result, it has nothing do with ragas
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[
            responseDict["retrieved_docs"][0]["page_content"],
            responseDict["retrieved_docs"][1]["page_content"],
            responseDict["retrieved_docs"][2]["page_content"]
        ],
        reference="23"  # expected result  "answer": "There are 23 articles in the course.  \n",
    )

    score = await context_recall.single_turn_ascore(sample)  # feed sinle turn constructor object
    print(score)
    assert score > 0.5  # test pass
