#service for pinecone storage and interfacing
from R4B.store.Embed import Embeddings
import pinecone
import os
import hashlib
import uuid
from dotenv import load_dotenv

load_dotenv('./.env')

class Storage:

    def __init__(self, initialize_indices_bool=True, pinecone_indices=["p-summary", "p-title", "p-tiktoken", "p-sentence"]):
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.pinecone_env = os.environ.get("PINECONE_ENV_NAME")

        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        self.pinecone_indices = pinecone_indices

        # self.pinecone_indices = ["p-sentence"] #For testing
        # self.processed_chunk_keys = ["Sentence"] #For Testing

        self.processed_chunk_keys = ["", "Title", "Tokens", "Sentence"]
        self.summary_index_name = "p-summary"
        self.embedding_object = Embeddings()

        # if initialize_indices_bool:
        #     print("Initializing Pinecone Indices")
        #     self.index_dict = self.init_pinecone_indices()
        #     print("Pinecone indices ready to use")


    def init_pinecone_indices(self):
        '''
            Initializes indices defined in self.pinecone_indices on pinecone.
        '''
        current_indices = pinecone.list_indexes()

        for index in self.pinecone_indices:
            if index not in current_indices:
                print("initializing "+ index)
                pinecone.create_index(name=index, dimension=384, metric="dotproduct", pod_type="s1")

        return {index:pinecone.Index(index) for index in self.pinecone_indices}
    

    def delete_pinecone_indices(self):
        '''
            Deletes all indices defined in self.pinecone_indices from pinecone
        '''
        current_indices = pinecone.list_indexes()

        for index in self.pinecone_indices:
            print("deleting "+index)
            if index in current_indices:
                pinecone.delete_index(index)


    def store_document_to_pinecone(self, processed_data, additional_metadata={}):
        '''
            Stores all possible chunks of a document
        '''
        source_id = str(uuid.uuid4())

        if "p-summary" in self.pinecone_indices:
            print("Storing Summary in the pinecone index p-summary")
            self.__store_summary([processed_data["Entire Document"]], source_id, "MAPREDUCE", additional_metadata)

        for index, chunk_key in zip(self.pinecone_indices, self.processed_chunk_keys):
            if chunk_key=="":
                continue
            print("Storing the "+ chunk_key+ " Chunk in the pinecone index "+ index)
            self.__store_chunks(index, processed_data[chunk_key], source_id, additional_metadata)


    def dense_query_pinecone(self, query_embed, index, context_size):
        '''
            Queries the given pinecone index using dense embeddings
        '''
        index = pinecone.Index(index)
        # query_embed = self.embedding_object.embed_dense(query)
        response = index.query(
                                vector = query_embed,
                                top_k = os.environ.get("TOP_K"),
                                include_values = True,
                                include_metadata=True
                            )
        id_list = [match["metadata"]["text"] for match in response["matches"]]
        return id_list
    

    def get_pinecone_index_dict(self):
        '''
            A getter function for the index dictionary
        '''
        return self.index_dict
        


    #################################### HELPER FUNCTIONS FOR STORAGE #################################################################
    def __store_summary(self, entire_text, source_id, chain_type, additional_metadata={}):
        '''
            Stores summary of the entire document in pinecone
        '''
        summary = self.embedding_object.create_summary(entire_text, chain_type)
        dense = self.embedding_object.embed_dense(summary)
        sparse = self.embedding_object.embed_sparse(summary)

        metadata = {
            "text": summary,
            "title": "",
            "pages": [],
            "source_id": source_id,
            "seq": -1,
            "hash": str(hashlib.sha512(bytes(summary, 'utf-8')).hexdigest())
        }

        for i in additional_metadata:
            metadata[i] = additional_metadata[i]

        upserts = [{
                    "id": str(source_id) + "__" + str(0) + "__" + str(1),
                    "sparse_values": sparse,
                    "values": dense,
                    "metadata": metadata
            }]

        self.index_dict["p-summary"].upsert(upserts)
        pass


    def __store_chunks(self, index, chunks, source_id, additional_metadata={}):
        '''
            Stores chunks to pinecone indices
        '''
        if len(chunks) == 0:
            return
        
        for i in range(0, len(chunks), 100):
            chunk_texts = [chunk.page_content for chunk in chunks]
            metadata_list = [chunk.metadata for chunk in chunks]
            seq_list = list(range(i, min(i+100, len(chunks))))
            chunks_dense = self.embedding_object.embed_dense_batch(chunk_texts)
            chunks_sparse = self.embedding_object.embed_sparse_batch(chunk_texts)

            upserts = [{
                "id": str(source_id) + "__" + str(seq) + "__" + str(len(chunks)), #change this to sourceID + _#_
                "sparse_values": sparse,
                "values": dense,
                "metadata": { **{
                    "text": text,
                    "title": metadata["title"] if "title" in metadata else "",
                    "pages": [str(p) for p in list(metadata["pages"])] if "pages" in metadata else [],
                    "source_id": source_id,
                    "seq": seq,
                    "hash": str(hashlib.sha512(bytes(text, 'utf-8')).hexdigest())
                }, **additional_metadata}
            } for (text, dense, sparse, metadata, seq) in zip(chunk_texts, chunks_dense, chunks_sparse, metadata_list, seq_list)]

            self.index_dict[index].upsert(upserts)