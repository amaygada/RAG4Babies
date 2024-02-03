from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import pinecone
import time
from R4B.Retriever.index_retrieve import IndexRetriever
from R4B.config import MULTIVECTOR_PROMPT_ALL, MULTIVECTOR_PROMPT_TIKTOKEN_SENTENCE, MULTIVECTOR_PROMPT_TITLE_SENTENCE, MULTIVECTOR_PROMPT_TITLE_TIKTOKEN
import os
from dotenv import load_dotenv

load_dotenv('./.env')
class MultiVectorRetrieval:

    def __init__(self, option="ALL", contextual_compression_bool=True, multi_query_bool=True):
        '''
            Options -> ["ALL", "TITLE_TIKTOKEN", "TITLE_SENTENCE", "TIKTOKEN_SENTENCE"]
        '''
        self.option = option
        self.contextual_compression_bool = contextual_compression_bool
        self.multi_query_bool = multi_query_bool
        self.prompt_choices = {
            "ALL": MULTIVECTOR_PROMPT_ALL, 
            "TITLE_TIKTOKEN": MULTIVECTOR_PROMPT_TITLE_TIKTOKEN, 
            "TITLE_SENTENCE": MULTIVECTOR_PROMPT_TITLE_SENTENCE, 
            "TIKTOKEN_SENTENCE": MULTIVECTOR_PROMPT_TIKTOKEN_SENTENCE
        }

    def retrieve(self, question, additional_instructions="None"):
        if self.option == "ALL":
            index_chunks = self.__get_index_retriever_results(question, sentence=True, title=True, tiktoken=True)
            time_start = time.time()
            print("Multi Vector Retrieval started")
            ret = self.__run_multivector_retriever_chain_all(question, index_chunks["Title"], index_chunks["Tiktoken"], index_chunks["Sentence"], index_chunks["Summary"], additional_instructions=additional_instructions)
            print("Time taken for MVR: ", time.time()-time_start)
            return ret
        elif self.option == "TITLE_TIKTOKEN":
            index_chunks = self.__get_index_retriever_results(question, sentence=False, title=True, tiktoken=True)
            time_start = time.time()
            print("Multi Vector Retrieval started")
            ret = self.__run_multivector_retriever_chain_title_tiktoken(question, index_chunks["Title"], index_chunks["Tiktoken"], index_chunks["Summary"], additional_instructions=additional_instructions)
            print("Time taken for MVR: ", time.time()-time_start)
            return ret
        elif self.option == "TITLE_SENTENCE":
            index_chunks = self.__get_index_retriever_results(question, sentence=True, title=True, tiktoken=False)
            time_start = time.time()
            print("Multi Vector Retrieval started")
            ret = self.__run_multivector_retriever_chain_title_sentence(question, index_chunks["Title"], index_chunks["Sentence"], index_chunks["Summary"], additional_instructions=additional_instructions)
            print("Time taken for MVR: ", time.time()-time_start)
            return ret
        elif self.option == "TIKTOKEN_SENTENCE":
            index_chunks = self.__get_index_retriever_results(question, sentence=True, title=False, tiktoken=True)
            time_start = time.time()
            print("Multi Vector Retrieval started")
            ret = self.__run_multivector_retriever_chain_tiktoken_sentence(question, index_chunks["Tiktoken"], index_chunks["Sentence"], index_chunks["Summary"], additional_instructions=additional_instructions)
            print("Time taken for MVR: ", time.time()-time_start)
            return ret

    def __run_multivector_retriever_chain_all(self, question, title_chunks, tiktoken_chunks, sentence_chunks, summary, additional_instructions="None"):
        prompt = PromptTemplate(template = self.prompt_choices[self.option], input_variables=["question", "title_chunks", "tiktoken_chunks", "sentence_chunks", "summary", "additional_instructions"])
        llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(question=question, title_chunks=title_chunks, tiktoken_chunks=tiktoken_chunks, sentence_chunks= sentence_chunks, summary=summary, additional_instructions=additional_instructions)
    
    def __run_multivector_retriever_chain_title_tiktoken(self, question, title_chunks, tiktoken_chunks, summary, additional_instructions="None"):
        prompt = PromptTemplate(template = self.prompt_choices[self.option], input_variables=["question", "title_chunks", "tiktoken_chunks", "summary", "additional_instructions"])
        llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(question=question, title_chunks=title_chunks, tiktoken_chunks=tiktoken_chunks, summary=summary, additional_instructions=additional_instructions)

    def __run_multivector_retriever_chain_title_sentence(self, question, title_chunks, sentence_chunks, summary, additional_instructions="None"):
        prompt = PromptTemplate(template = self.prompt_choices[self.option], input_variables=["question", "title_chunks", "sentence_chunks", "summary", "additional_instructions"])
        llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(question=question, title_chunks=title_chunks, sentence_chunks=sentence_chunks, summary=summary, additional_instructions=additional_instructions)
    
    def __run_multivector_retriever_chain_tiktoken_sentence(self, question, tiktoken_chunks, sentence_chunks, summary, additional_instructions="None"):
        prompt = PromptTemplate(template = self.prompt_choices[self.option], input_variables=["question", "tiktoken_chunks", "sentence_chunks", "summary", "additional_instructions"])
        llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(question=question, tiktoken_chunks=tiktoken_chunks, sentence_chunks= sentence_chunks, summary=summary, additional_instructions=additional_instructions)
    
    def __get_summary_from_source(self, source):
        print("Retrieving Summary!")
        index = pinecone.Index("p-summary")
        key = source[0]+ "__" + str(0) + "__" + str(1)
        return index.fetch([key])["vectors"][key]["metadata"]["text"]
    
    def __get_index_retriever_results(self, question, sentence=False, title=False, tiktoken=False):
        print("INDEX RETRIEVAL STARTED")
        time_start = time.time()

        sources = ""
        chunks = {}

        if sentence:
            i = IndexRetriever(os.environ.get("SENTENCE_INDEX"), contextual_compression_bool=self.contextual_compression_bool, multi_query_bool=self.multi_query_bool)
            sc = i.retrieve(question)
            sources = sc[0].metadata["sources"]
            sentence_chunks = ""
            for k in range(len(sc)):
                sentence_chunks += str(k+1)+". "+sc[k].page_content+" \n\n"
            chunks["Sentence"] = sentence_chunks
        
        if tiktoken:
            i = IndexRetriever(os.environ.get("TIKTOKEN_INDEX"), contextual_compression_bool=self.contextual_compression_bool, multi_query_bool=self.multi_query_bool)
            tc = i.retrieve(question)
            sources = tc[0].metadata["sources"]
            tiktoken_chunks = ""
            for k in range(len(tc)):
                tiktoken_chunks+str(k+1)+". "+tc[k].page_content+" \n\n"
            chunks["Tiktoken"] = tiktoken_chunks

        if title:
            i = IndexRetriever(os.environ.get("TITLE_INDEX"), contextual_compression_bool=self.contextual_compression_bool, multi_query_bool=self.multi_query_bool)
            tic = i.retrieve(question)
            sources = tic[0].metadata["sources"]
            title_chunks = ""
            for k in range(len(tic)):
                title_chunks += str(k+1)+". "+tic[k].page_content+" \n\n"
            chunks["Title"] = title_chunks

        summary = self.__get_summary_from_source(sources)
        chunks["Summary"] = summary

        print("Time taken for Index R: ", time.time()-time_start)


        return chunks