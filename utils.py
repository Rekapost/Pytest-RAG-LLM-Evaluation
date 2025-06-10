import json
from pathlib import Path
import requests

# question from testset can be written as  user_input=test_data["eval_sample"]["user_input"]
def get_llm_response(test_data):
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": test_data["question"],
                                     "chat_history": [
                                     ]
                                 }
                                 ).json()
    print(responseDict)
    return responseDict

def load_json_test_data(filename):
    # path of parent
    project_directory = Path(__file__).parent.absolute()
    test_data_path = project_directory / "testdata" / filename
    #test_data_path = r"C:\Users\nreka\PycharmProjects\Pytest-RAG-LLM-Evaluation\testdata\testdata-recall.json"
    with open(test_data_path) as f:  # open file,   file is refereed as object f
        return json.load(f)
