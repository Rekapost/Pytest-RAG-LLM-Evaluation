import os

import pytest
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
import nltk

#LLM - 3 docs
# This is not actual Automation testing, it is about generating testdata, we are not calculating metrics , not asserting
# commenting @pytest.mark.asyncio and async def test_data_Creation(): and run as standalone test, consider this as utility file , so that u can run and generate testdata
# so no pytest
os.environ["RAGAS_APP_TOKEN"] = "apt.4a91-0b082a-f0bb2"
os.environ["OPENAI_API_KEY"] = "sk-proj-csXhU7UeVqgTXfN3oxFbhTFscTu2or7pzFJxTYA"
api_key = os.environ["OPENAI_API_KEY"]

nltk.data.path.append("C://Users//nreka//docker//Testing//RAG-LLM//nltk_data")
llm = ChatOpenAI(model="gpt-3.5-turbo",
                 temperature=0,
                 api_key=SecretStr(api_key)
                 )  # llm object
langchain_llm = LangchainLLMWrapper(
    llm)  # class, llm object is converted to ragas understantable format by using wrapper
embedded = OpenAIEmbeddings()
loader = DirectoryLoader(
    path="C://Users//nreka//docker//Testing//RAG-LLM//LLM+Evaluation_Resources//LLM Evaluation_Resources//fs11//",
    glob="**/*.docx",
    loader_cls=UnstructuredWordDocumentLoader
    #loader_cls=Docx2txtLoader
)
docs = loader.load()
for i, doc in enumerate(docs):
    print(f"\n--- Document {i + 1} ---")
    print(doc.page_content[:500])

if not docs:
    raise ValueError("No documents were loaded. Check your path or file contents.")

generate_embeddings = LangchainEmbeddingsWrapper(embedded)
generator = TestsetGenerator(llm=langchain_llm, embedding_model=generate_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=20)  # testcases are generated
print(dataset.to_list())  # converting dataset to list format
dataset.upload()  # upload dataset to ragas framework
