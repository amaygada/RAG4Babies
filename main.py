#Is the main file from where everything is accessed

from R4B.FileLoaders.pipeline import DataPipeline
from R4B.store.Storage import Storage
from R4B.store.Summary import Summarize, parallelize_summary
from R4B.Retriever.multivector_retriever import MultiVectorRetrieval
from R4B.Retriever.retriever import Retriever
import re

from langchain.docstore.document import Document
import torch

from R4B.Retriever.index_retrieve import IndexRetriever

if __name__ == "__main__":

    # print(os.environ.get("TOP_K"))

    # torch.cuda.set_device(torch.device("cuda:0"))
    
    # LOAD DATA
    p = DataPipeline()
    # pp = p.process_document("p.pdf", 384, 20)
    # print(len(pp["Entire Document"].page_content))
    # tokens = re.split(r'(?<=[.!?])\s+', pp["Entire Document"].page_content)
    # print(tokens[:10])
    # print(len(tokens))
    # print("bb")
    # s = Summarize()
    # print(parallelize_summary(pp["Entire Document"].page_content))
    # print(s.summarize([pp["Entire Document"]], "MAPREDUCE"))
    
    
    # STORE DATA IN PINECONE
    # s = Storage()
    # s.delete_pinecone_indices()
    # s.store_document_to_pinecone(pp)

    # response = s.dense_query_pinecone("What is limited direct execution?", "p-title", 0)
    # print(response["matches"])
    # for r in response:
    #     print(r)
    #     print("#########")
    #     print()


    # mm = MultiVectorRetrieval(option="TIKTOKEN_SENTENCE", contextual_compression_bool=False, multi_query_bool=False)
    # Question = "What is the return to trap instruction that is mentioned in the OS text book Three Easy Pieces?"
    # temp = """
    #         Give a catchy heading to the response. Write as if you are writing a conversation between a student and a professor. The professor is explaining the concept to the student.
    #     """
    

    # ans = mm.retrieve(Question, additional_instructions=temp)
    # print(end="\n\n")
    # print("########################################################################")
    # print(ans)

    # temp = """
    #     Explain the concept to me on the basis of the retrieved chunks. Do not use additional context.
    # """
    # r = Retriever()
    # ans = r.index_retrieve(Question, "p-title", additional_instructions=temp)
    # print(ans)