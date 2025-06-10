import os
import pytest
from ragas.evaluation import EvaluationDataset, evaluate
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy, FactualCorrectness
from utils import load_json_test_data, get_llm_response

os.environ["RAGAS_APP_TOKEN"] = "apt.4a91-06bc6492eef4-d52f-88bb-1eeb082a-f0bb2"


@pytest.mark.parametrize("getData", load_json_test_data("testdata-relavance-factual.json"), indirect=True)
@pytest.mark.asyncio
async def test_relevancy_factual(llm_wrapper, getData):
    metrics = [ResponseRelevancy(llm=llm_wrapper),
               FactualCorrectness(llm=llm_wrapper)]
    eval_dataset = EvaluationDataset([getData])
    results = evaluate(dataset=eval_dataset, metrics=metrics)
    #results = evaluate(dataset=eval_dataset)
    print(results)
    print(results["answer_relevancy"])
    #assert
    results.upload()


@pytest.fixture
def getData(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    sample = SingleTurnSample(
        user_input=test_data["question"],
        response=responseDict["answer"],
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")],
        reference=test_data["reference"]
    )
    return sample
