from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv('./.env')
# from config import CHAT_MODEL


class MultiQuery:

    def __init__(self, num_versions=3):

        if (num_versions>9) or (num_versions<1):
            raise RuntimeError("num_versions should be in the range (0,9]")
        
        self.template = """You are an AI language model assistant. Your task is to generate {num_versions} 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}"""

        self.num_versions = num_versions


    def generate_multiple_queries(self, question):
        prompt = PromptTemplate(input_variables=["question", "num_versions"], template = self.template)
        llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        response = llm_chain.apply([{"question": question, "num_versions": self.num_versions}])[0]['text']
        return [question] + [r[3:] for r in response.split("\n")]

