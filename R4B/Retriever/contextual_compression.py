from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv('./.env')
# from config import CHAT_MODEL

class ContextualCompression:
    
    def __init__(self, base_retriever, compressor_mode):
        self.base_retriever = base_retriever
        self.compressor_mode = compressor_mode # can be either of [Filter, Extractor]

    def compressed_retrieval(self, query):
        compressor = self._get_compressor()
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=self.base_retriever)
        return compression_retriever.get_relevant_documents(query)
    
    def _get_compressor(self):
        llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
        if self.compressor_mode == "Extractor":
            return LLMChainExtractor.from_llm(llm)
        elif self.compressor_mode == "Filter":
            return LLMChainFilter.from_llm(llm)
        else:
            raise RuntimeError("Invalid Compressor Mode")

