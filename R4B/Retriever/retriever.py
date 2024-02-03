from R4B.Retriever.multivector_retriever import MultiVectorRetrieval
from R4B.Retriever.index_retrieve import IndexRetriever
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import pinecone
from R4B.config import INDEX_PROMPT_SENTENCE, INDEX_PROMPT_TIKTOKEN, INDEX_PROMPT_TITLE, SUMMARY_INDEX, TITLE_INDEX, SENTENCE_INDEX, TIKTOKEN_INDEX
import os
from dotenv import load_dotenv

load_dotenv('./.env')

class Retriever:

    def __init__(self):
        self.index_prompt_options = {
            os.environ.get("SENTENCE_INDEX"): INDEX_PROMPT_SENTENCE,
            os.environ.get("TIKTOKEN_INDEX"): INDEX_PROMPT_TIKTOKEN,
            os.environ.get("TITLE_INDEX"): INDEX_PROMPT_TITLE
        }

    def multi_vector_retrieve(self, question, option="ALL", contextual_compression_bool=False, multi_query_bool=True, additional_instructions="None"):
        mm = MultiVectorRetrieval(option=option, contextual_compression_bool=contextual_compression_bool, multi_query_bool=multi_query_bool)
        return mm.retrieve(question, additional_instructions)
    
    def index_retrieve(self, question, index_name, contextual_compression_bool=True, multi_query_bool=True, additional_instructions="None"):
        chunks, summary = self.__get_index_chunks(question, index_name, contextual_compression_bool=contextual_compression_bool, multi_query_bool=multi_query_bool)
        prompt = PromptTemplate(template = self.index_prompt_options[index_name], input_variables=["question", "chunks", "summary", "additional_instructions"])
        llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(question=question, chunks= chunks, summary=summary, additional_instructions=additional_instructions)
    
    def __get_summary_from_source(self, source):
        # print("Retrieving Summary!")
        index = pinecone.Index(SUMMARY_INDEX)
        key = source[0]+ "__" + str(0) + "__" + str(1)
        return index.fetch([key])["vectors"][key]["metadata"]["text"]
    
    def __get_index_chunks(self, question, index_name, contextual_compression_bool, multi_query_bool):
        sources = ""
        chunks = {}

        i = IndexRetriever(index_name, contextual_compression_bool=contextual_compression_bool, multi_query_bool=multi_query_bool)
        sc = i.retrieve(question)

        if len(sc) == 0:
            print("Nothing found in index: ", index_name)
            return("NA", "NA")


        sources = sc[0].metadata["sources"]
        c = ""
        for k in range(len(sc)):
            c += str(k+1)+". "+sc[k].page_content+" \n\n"
        chunks["chunk"] = c

        summary = self.__get_summary_from_source(sources)
        
        return (chunks["chunk"], summary)