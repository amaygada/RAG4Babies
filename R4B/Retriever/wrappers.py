from langchain.schema.retriever import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
from R4B.Retriever.retrieve_helper import RetrieverHelper
# from config import CONTEXT_WINDOW
import os
from dotenv import load_dotenv

load_dotenv('./.env')

CONTEXT_WINDOW = os.environ.get("CONTEXT_WINDOW")

class  RetrieverWrapperSingleQuery(BaseRetriever):
    index = ""

    def __init__(self):
        super().__init__()

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # print("Getting RD 1")
        r = RetrieverHelper(context_size=CONTEXT_WINDOW, multi_query=False)
        (sources, res) = r.get_query_result(query, self.index)
        ll = []
        for rr in res:
            ll.append(Document(page_content=rr, metadata={"sources": list(sources)}))
        return ll
     
    def set_index(self, ind):
        self.index = ind
        
    def get_index(self):
        return self.index



class  RetrieverWrapperMultiQuery(BaseRetriever):
    index = ""

    def __init__(self):
        super().__init__()
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # print("Getting RD 2")
        r = RetrieverHelper(context_size=CONTEXT_WINDOW, multi_query=True)
        (sources, res) = r.get_query_result(query, self.index)
        ll = []
        for rr in res:
            ll.append(Document(page_content=rr, metadata={"sources": list(sources)}))
        return ll
    
    def set_index(self, ind):
        self.index = ind
    
    def get_index(self):
        return self.index
