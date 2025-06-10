import pytest
from ragas.messages import HumanMessage, AIMessage
from ragas import MultiTurnSample
from ragas.metrics import TopicAdherenceScore
from utils import load_json_test_data, get_llm_response


@pytest.fixture
def getdata():
    #test_data = request.param  # to capture json data
    #responseDict = get_llm_response(test_data)  # response
    conversation = [
        HumanMessage(content="How many articles are there in the Selenium Webdriver python course?"),
        AIMessage(content="There are 23 articles in the Selenium WebDriver Python course."),
        HumanMessage(content="How many downloadable resources are there in this course?"),
        AIMessage(content="There are 9 downloadable resources in the course.")
    ]

    reference = [
        """
        The AI should:
        1. Give results related to the selenium webdriver python course.
        2. There are 23 articles and 9 downloadable resources in the course.
        """]

    sample = MultiTurnSample(
        user_input=conversation,
        reference_topics=reference
    )
    return sample


#@pytest.mark.parametrize("getData", load_json_test_data("testData-Faith.json"), indirect=True)
@pytest.mark.asyncio
async def test_topicAdherence(llm_wrapper, getdata):
    print("LLM Wrapper:", llm_wrapper)
    topicadherencescore = TopicAdherenceScore(llm=llm_wrapper)
    print("Sample:", getdata.dict())
    print("LLM Type:", type(llm_wrapper))

    score = await topicadherencescore.multi_turn_ascore(getdata)
    print(score)
    assert score > 0.8
