#service to embed texts into vector embeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from pinecone_text.sparse import SpladeEncoder
from R4B.store.Summary import Summarize, parallelize_summary
# from config import DENSE_EMBEDDER_MODEL
import os
from dotenv import load_dotenv, dotenv_values

load_dotenv('./.env')

class Embeddings:
    def __init__(self):
        self.dense_model = self.load_dense_model()
        self.sparse_model = self.load_sparse_model()
        self.summary_object = Summarize()


    def load_dense_model(self):
        model_name = os.environ.get("DENSE_EMBEDDER_MODEL")
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return hf
    
    def load_sparse_model(self):
        return SpladeEncoder()
    
    def embed_dense(self, text):
        return self.dense_model.embed_documents([text])[0]
    
    def embed_sparse(self, text):
        return self.sparse_model.encode_documents([text])[0]
    
    def embed_dense_batch(self, texts):
        return self.dense_model.embed_documents(texts)
        
    def embed_sparse_batch(self, texts):
        return self.sparse_model.encode_documents(texts)
    
    def create_summary(self, entire_text, chain_type="MAPREDUCE"):
        # return self.summary_object.summarize(entire_text, chain_type)
        return parallelize_summary(entire_text)
