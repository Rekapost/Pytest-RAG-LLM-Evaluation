import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from utils import load_json_test_data, get_llm_response

# to feed data for fixtures so it will create single turn object for us
# to create data , we need data json file
# response and relevant docs , API will give
# as user , input we send only question what we need
# wrapper to call json load_json_test_data

@pytest.fixture
def getData(request):
    test_data = request.param  # to capture json data
    responseDict = get_llm_response(test_data)  # response
    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")]
        #responseDict["retrieved_docs"][0]["page_content"].......
    )
    return sample
@pytest.mark.parametrize("getData", load_json_test_data("testData-Faith.json"), indirect=True)
@pytest.mark.asyncio
async def test_faithfulness(llm_wrapper, getData):
    #Faithfulness is class in RAGAS , to get mertic of faithfulness core
    faithful = Faithfulness(llm=llm_wrapper)
    score = await faithful.single_turn_ascore(getData)
    print(score)
    assert score > 0.8
