from R4B.Retriever.multi_query import MultiQuery
from R4B.store.Storage import Embeddings
import pinecone
import os
from dotenv import load_dotenv

load_dotenv('./.env')
# from config import TOP_K


class RetrieverHelper:
    def __init__(self, context_size=1, multi_query=True):
        self.context_size = context_size
        self.multi_query_bool = multi_query
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.pinecone_env = os.environ.get("PINECONE_ENV_NAME")
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)

        self.multi_query_object = MultiQuery(num_versions=3)
        self.embedding_object = Embeddings()


    def get_query_result(self, question, index):
        questions = []
        if self.multi_query_bool:
            questions = self.__get_multiple_queries(question)
        else:
            questions = [question]

        query_embedding_list = self.__get_and_store_query_embeddings(questions)
        (sources, res) = self.__query_results(index, query_embedding_list)
        return (sources, res)


    ################################ PRIVATE HELPER FUNCTIONS TO AID RETRIEVAL ###########################################################

    def __dense_query_pinecone(self, query_embed, index, context_size):
        '''
            Queries the given pinecone index using dense embeddings
        '''
        response = index.query(
                                vector = query_embed,
                                top_k = int(os.environ.get("TOP_K")),
                                include_values = True,
                                include_metadata=True
                                )

        ids = [match["id"] for match in response["matches"]]
        # id_list = [match["metadata"]["text"] for match in response["matches"]]
        return ids
    
    
    def __get_multiple_queries(self, question):
        return self.multi_query_object.generate_multiple_queries(question) 


    def __get_and_store_query_embeddings(self, query_list):
        return self.embedding_object.embed_dense_batch(query_list)


    def __query_results(self, index, query_embedding_list):
        result = set()
        index = pinecone.Index(index)
        for query in query_embedding_list:
            query_res = self.__dense_query_pinecone(query, index, self.context_size)
            for q in query_res:
                result.add(q)

        id_list = []
        grouped_ids = []
        source_set = set()
        for id in result:
            temp = []
            split = id.split("__")
            source = split[0]
            seq = int(split[1])
            limit = int(split[2])
            temp.append(id)
            if seq!=0:
                temp.append(source + "__" + str(seq-1) + "__" + str(limit))
            if seq!=limit-1:
                temp.append(source + "__" + str(seq+1) + "__" + str(limit))
            grouped_ids.append(temp)
            id_list+=temp
            source_set.add(source)

        result = []
        chunks = index.fetch(list(set(id_list)))["vectors"]
        for g in grouped_ids:
            s = ""
            for i in g:
                s+=chunks[i]["metadata"]["text"]
            result.append(s)

        return (source_set, result) 