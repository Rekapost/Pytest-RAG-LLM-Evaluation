import json
import requests

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

def load_json_test_data():
    test_data_path = r"C:\Users\nreka\PycharmProjects\Pytest-RAG-LLM-Evaluation\testdata\testdata.json"
    with open(test_data_path) as f:    # open file,   file is refereed as object f
        return json.load(f)