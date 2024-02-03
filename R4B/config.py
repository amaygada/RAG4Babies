#LLM
CHAT_MODEL="gpt-4-1106-preview"  #all LLM chains use this chat model for retrieval


#PINECONE QUERYING
CONTEXT_WINDOW = 1  # When an index is queried, we fetch a window of this size before and after the retrieved chunk
TOP_K = 3           # The pinecone query will return top K closest chunks
TITLE_INDEX = "p-title"
TIKTOKEN_INDEX = "p-tiktoken"
SENTENCE_INDEX = "p-sentence"
SUMMARY_INDEX = "p-summary"
DENSE_EMBEDDER_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

#STORAGE
SENTENCE_PER_TASK_PARALLEL_SUMMARY = 120


#PROMPTS
INDEX_PROMPT_TITLE = """ Given the following question (look below), the corresponding retrieved chunks from one store, and the summary of the text as context, please provide a comprehensive and informative answer that synthesizes the information from all the retrieved chunks and the summary, while considering potential overlaps and utilizing learned knowledge from your training to enhance the response. Note that the answer should be to the QUESTION POSED. 

                            Please be aware that the retrieved chunks from different stores may contain overlapping information.

                            You can utilize learned knowledge from your training to enhance the response, particularly for open-ended questions. Whether the question is of an open ended nature is up to you to decide.

                            As you should remember, we use chunks from the TITLE CHUNK STORE. The TITLE CHUNK STORE contains smartly chunked title and content pairs. All these chunks are created from the same factual document that is necessary for answering factually to the question posed.

                            Efforts have been taken to minimize overlaps from the different chunk stores. Efforts have also been taken to ensure that the retrieved chunks are relevant to the question. However, sometimes this may not be the case. So it is upon YOU to figure what is important and what is not important and form an answer accordingly. 

                            Note that the response should only have the answer to the question that is formed by abiding to the guidelines described above.


                            QUESTION: {question}

                            ADDITIONAL INSTRUCTIONS: {additional_instructions}

                            RETRIEVED CHUNKS FROM THE TITLE CHUNK STORE:
                                {chunks}
                                
                            SUMMARY: 
                                {summary}

"""

INDEX_PROMPT_TIKTOKEN = """ Given the following question (look below), the corresponding retrieved chunks from a store, and the summary of the text as context, please provide a comprehensive and informative answer that synthesizes the information from all the retrieved chunks and the summary, while considering potential overlaps and utilizing learned knowledge from your training to enhance the response. Note that the answer should be to the QUESTION POSED. 

                            Please be aware that the retrieved chunks from different stores may contain overlapping information.

                            You can utilize learned knowledge from your training to enhance the response, particularly for open-ended questions. Whether the question is of an open ended nature is up to you to decide.

                            As you should remember, we use chunks from the TIKTOKEN CHUNK STORE. TIKTOKEN CHUNK STORE contains chunks created based on a predefined token size. All these chunks are created from the same factual document that is necessary for answering factually to the question posed.

                            Efforts have been taken to minimize overlaps from the different chunk stores. Efforts have also been taken to ensure that the retrieved chunks are relevant to the question. However, sometimes this may not be the case. So it is upon YOU to figure what is important and what is not important and form an answer accordingly. 

                            Note that the response should only have the answer to the question that is formed by abiding to the guidelines described above.


                            QUESTION: {question}

                            ADDITIONAL INSTRUCTIONS: {additional_instructions}

                            RETRIEVED CHUNKS FROM THE TIKTOKEN CHUNK STORE:
                                {chunks}
                                
                            SUMMARY: 
                                {summary}

"""

INDEX_PROMPT_SENTENCE = """ Given the following question (look below), the corresponding retrieved chunks from a stores, and the summary of the text as context, please provide a comprehensive and informative answer that synthesizes the information from all the retrieved chunks and the summary, while considering potential overlaps and utilizing learned knowledge from your training to enhance the response. Note that the answer should be to the QUESTION POSED. 

                            Please be aware that the retrieved chunks from different stores may contain overlapping information.

                            You can utilize learned knowledge from your training to enhance the response, particularly for open-ended questions. Whether the question is of an open ended nature is up to you to decide.

                            As you should remember, we use chunks from the SENTENCE CHUNK STORE. The SENTENCE CHUNK STORE groups a few sentences together to form chunks of the text. All these chunks are created from the same factual document that is necessary for answering factually to the question posed.

                            Efforts have been taken to minimize overlaps from the different chunk stores. Efforts have also been taken to ensure that the retrieved chunks are relevant to the question. However, sometimes this may not be the case. So it is upon YOU to figure what is important and what is not important and form an answer accordingly. 

                            Note that the response should only have the answer to the question that is formed by abiding to the guidelines described above.


                            QUESTION: {question}

                            ADDITIONAL INSTRUCTIONS: {additional_instructions}
                                
                            RETRIEVED CHUNKS FROM THE SENTENCE CHUNK STORE:
                                {chunks}
                                
                            SUMMARY: 
                                {summary}

"""

MULTIVECTOR_PROMPT_ALL = """ Given the following question (look below), the corresponding retrieved chunks from different stores, and the summary of a text as context, please provide a comprehensive and informative answer that synthesizes the information from all the retrieved chunks and the summary, while considering potential overlaps and utilizing learned knowledge from your training to enhance the response. Note that the answer should be to the QUESTION POSED. 

                            Please be aware that the retrieved chunks from different stores may contain overlapping information.

                            You can utilize learned knowledge from your training to enhance the response, particularly for open-ended questions. Whether the question is of an open ended nature is up to you to decide.

                            As you should remember, we use chunks from 3 different stores, the TITLE CHUNK STORE, TIKTOKEN CHUNK STORE, AND SENTENCE CHUNK STORE. The TITLE CHUNK STORE contains smartly chunked title and content pairs. TIKTOKEN CHUNK STORE contains chunks created based on a predefined token size. The SENTENCE CHUNK STORE groups a few sentences together to form chunks of the text. All these chunks are created from the same factual document that is necessary for answering factually to the question posed.

                            Efforts have been taken to minimize overlaps from the different chunk stores. Efforts have also been taken to ensure that the retrieved chunks are relevant to the question. However, sometimes this may not be the case. So it is upon YOU to figure what is important and what is not important and form an answer accordingly. 

                            Note that the response should only have the answer to the question that is formed by abiding to the guidelines described above.


                            QUESTION: {question}

                            
                            ADDITIONAL INSTRUCTIONS: {additional_instructions}


                            RETRIEVED CHUNKS FROM THE TITLE CHUNK STORE:
                                {title_chunks}

                            RETRIEVED CHUNKS FROM THE TIKTOKEN CHUNK STORE:
                                {tiktoken_chunks}
                                
                            RETRIEVED CHUNKS FROM THE SENTENCE CHUNK STORE:
                                {sentence_chunks}
                                
                            SUMMARY: 
                                {summary}          
                    """

MULTIVECTOR_PROMPT_TITLE_TIKTOKEN = """ Given the following question (look below), the corresponding retrieved chunks from different stores, and the summary of a text as context, please provide a comprehensive and informative answer that synthesizes the information from all the retrieved chunks and the summary, while considering potential overlaps and utilizing learned knowledge from your training to enhance the response. Note that the answer should be to the QUESTION POSED. 

                            Please be aware that the retrieved chunks from different stores may contain overlapping information.

                            You can utilize learned knowledge from your training to enhance the response, particularly for open-ended questions. Whether the question is of an open ended nature is up to you to decide.

                            As you should remember, we use chunks from 2 different stores, the TITLE CHUNK STORE, AND TIKTOKEN CHUNK STORE. The TITLE CHUNK STORE contains smartly chunked title and content pairs. TIKTOKEN CHUNK STORE contains chunks created based on a predefined token size. All these chunks are created from the same factual document that is necessary for answering factually to the question posed.

                            Efforts have been taken to minimize overlaps from the different chunk stores. Efforts have also been taken to ensure that the retrieved chunks are relevant to the question. However, sometimes this may not be the case. So it is upon YOU to figure what is important and what is not important and form an answer accordingly. 

                            Note that the response should only have the answer to the question that is formed by abiding to the guidelines described above.


                            QUESTION: {question}

                            
                            ADDITIONAL INSTRUCTIONS: {additional_instructions}

                            
                            RETRIEVED CHUNKS FROM THE TITLE CHUNK STORE:
                                {title_chunks}

                            RETRIEVED CHUNKS FROM THE TIKTOKEN CHUNK STORE:
                                {tiktoken_chunks}
                                
                            SUMMARY: 
                                {summary}

"""

MULTIVECTOR_PROMPT_TITLE_SENTENCE = """Given the following question (look below), the corresponding retrieved chunks from different stores, and the summary of a text as context, please provide a comprehensive and informative answer that synthesizes the information from all the retrieved chunks and the summary, while considering potential overlaps and utilizing learned knowledge from your training to enhance the response. Note that the answer should be to the QUESTION POSED. 

                            Please be aware that the retrieved chunks from different stores may contain overlapping information.

                            You can utilize learned knowledge from your training to enhance the response, particularly for open-ended questions. Whether the question is of an open ended nature is up to you to decide.

                            As you should remember, we use chunks from 2 different stores, the TITLE CHUNK STORE, AND SENTENCE CHUNK STORE. The TITLE CHUNK STORE contains smartly chunked title and content pairs. The SENTENCE CHUNK STORE groups a few sentences together to form chunks of the text. All these chunks are created from the same factual document that is necessary for answering factually to the question posed.

                            Efforts have been taken to minimize overlaps from the different chunk stores. Efforts have also been taken to ensure that the retrieved chunks are relevant to the question. However, sometimes this may not be the case. So it is upon YOU to figure what is important and what is not important and form an answer accordingly. 

                            Note that the response should only have the answer to the question that is formed by abiding to the guidelines described above.


                            QUESTION: {question}

                            
                            ADDITIONAL INSTRUCTIONS: {additional_instructions}


                            RETRIEVED CHUNKS FROM THE TITLE CHUNK STORE:
                                {title_chunks}
                                
                            RETRIEVED CHUNKS FROM THE SENTENCE CHUNK STORE:
                                {sentence_chunks}
                                
                            SUMMARY: 
                                {summary}

"""

MULTIVECTOR_PROMPT_TIKTOKEN_SENTENCE = """Given the following question (look below), the corresponding retrieved chunks from different stores, and the summary of a text as context, please provide a comprehensive and informative answer that synthesizes the information from all the retrieved chunks and the summary, while considering potential overlaps and utilizing learned knowledge from your training to enhance the response. Note that the answer should be to the QUESTION POSED. 

                            Please be aware that the retrieved chunks from different stores may contain overlapping information.

                            You can utilize learned knowledge from your training to enhance the response, particularly for open-ended questions. Whether the question is of an open ended nature is up to you to decide.

                            As you should remember, we use chunks from 2 different stores, the TIKTOKEN CHUNK STORE, AND SENTENCE CHUNK STORE. The TITLE CHUNK STORE contains smartly chunked title and content pairs. TIKTOKEN CHUNK STORE contains chunks created based on a predefined token size. The SENTENCE CHUNK STORE groups a few sentences together to form chunks of the text. All these chunks are created from the same factual document that is necessary for answering factually to the question posed.

                            Efforts have been taken to minimize overlaps from the different chunk stores. Efforts have also been taken to ensure that the retrieved chunks are relevant to the question. However, sometimes this may not be the case. So it is upon YOU to figure what is important and what is not important and form an answer accordingly. 

                            Note that the response should only have the answer to the question that is formed by abiding to the guidelines described above.

                            QUESTION: {question}

                            ADDITIONAL INSTRUCTIONS: {additional_instructions}

                            RETRIEVED CHUNKS FROM THE TIKTOKEN CHUNK STORE:
                                {tiktoken_chunks}
                                
                            RETRIEVED CHUNKS FROM THE SENTENCE CHUNK STORE:
                                {sentence_chunks}
                                
                            SUMMARY: 
                                {summary}

"""

PARALLEL_MAP_TEMPLATE_SUMMARY = """The following is a piece of a larger documnt 
                                            
    {text} 
    
    Based on this piece of text, please identify the main themes. The main themes should be long enough to allow all important
    points and themes to be convered. Along with main themes, also give additional context for each. 
    The output should be propotional to the size of the input. (Not more than 4/5 sentences)
    
    Helpful Answer:"""

PARALLEL_REDUCE_TEMPLATE_SUMMARY = """The following is set of summaries:

        {doc_summaries}

        Take these and distill it into a final, consolidated summary of the main themes, along with context and details for each main theme.
        Output a paragraph of the summary and make sure to not miss anything important.
        Also do not lose sense of sections. Give the summary section wise, maintaining that isolation between ideas and content.
        The summary should be long enough to allow all important points and themes to be covered. Combine themes and ideas if you feel like and create a summary such that it does not become too big also.

        Helpful Answer:"""