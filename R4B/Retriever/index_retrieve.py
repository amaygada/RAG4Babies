from R4B.Retriever.wrappers import RetrieverWrapperMultiQuery, RetrieverWrapperSingleQuery
from R4B.Retriever.contextual_compression import ContextualCompression
from datetime import datetime

class IndexRetriever:
    def __init__(self, index, multi_query_bool = True, contextual_compression_bool = True, compressor_mode="Extractor"):
        self.multi_query_bool = multi_query_bool
        self.contextual_compression_bool = contextual_compression_bool
        self.pinecone_index = index
        if multi_query_bool:
            self.base_retriever = RetrieverWrapperMultiQuery()
        else:
            self.base_retriever = RetrieverWrapperSingleQuery()
        self.base_retriever.set_index(index)

        if(contextual_compression_bool):
            self.contextual_compression_obj = ContextualCompression(self.base_retriever, compressor_mode=compressor_mode)

    
    def retrieve(self, query):
        print("Retrieving chunks from: ", self.pinecone_index)
        if self.contextual_compression_bool:
            return self.contextual_compression_obj.compressed_retrieval(query)
        else:
            return self.base_retriever.get_relevant_documents(query)
