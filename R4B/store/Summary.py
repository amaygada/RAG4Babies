
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
import time
import re
import threading
from R4B.config import PARALLEL_MAP_TEMPLATE_SUMMARY, PARALLEL_REDUCE_TEMPLATE_SUMMARY

import os
from dotenv import load_dotenv

load_dotenv('./.env')


class Summarize:
    def __init__(self, map_reduce_divisions=2):
        self.prompt_template = "Can you provide a comprehensive summary of the given text? " \
                              "The summary should cover all the key points and main ideas presented " \
                              "in the original text, while also condensing the information into a concise " \
                              "and easy-to-understand format. Please ensure that the summary includes relevant " \
                              "details and examples that support the main ideas, while avoiding any unnecessary " \
                              "information or repetition. The length of the summary should be appropriate for the" \
                              " length and complexity of the original text, providing a clear and accurate " \
                              "overview without omitting any important information."\
                              "Begin! "\
                              """ "{text}" """ \
                              "CONCISE SUMMARY: "
        
        self.refine_template = (
                                "Your job is to produce a final summary\n"
                                "We have provided an existing summary up to a certain point: {existing_answer}\n"
                                "You have the opportunity to refine the existing summary"
                                "with some more context below.\n"
                                "------------\n"
                                "{text}\n"
                                "------------\n"
        )

        self.map_template = """The following is a set of documents 
                                            
                                            {docs} 
                                            
                                            Based on this list of docs, please identify the main themes. The main themes should be long enough to allow all important
                                            points and themes to be convered. Along with main themes, also give additional context for each. 
                                            I do not care if the output is slightly longer. 
                                            
                                            Helpful Answer:"""
        
        self.reduce_template = """The following is set of summaries:

                                               {doc_summaries}

                                               Take these and distill it into a final, consolidated summary of the main themes, along with context and details for each main theme.
                                               Output a paragraph of the summary and make sure to not miss anything important.
                                               Also do not lose sense of sections. Give the summary section wise, maintaining that isolation between ideas and content.
                                               Lastly, I do not care if the output is slightly longer. The summary should be long enough to allow all important points and themes to be covered.
                                    
                                              Helpful Answer:"""

        self.map_reduce_divisions = map_reduce_divisions

        self.maps = [] #stores results from parallel summary tasks


    def __run_stuff_chain(self, entire_text):
        '''
            Create Stuffed LLM Chain for summarizing text
        '''
        prompt = PromptTemplate.from_template(self.prompt_template)
        llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, document_variable_name="text")
        return chain.run(entire_text)
    
    

    def __run_refined_chain(self, entire_text):
        '''
            Creates Refined LLM chain for summarizing text
        '''
        prompt = PromptTemplate.from_template(self.prompt_template)
        llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
    
        refine_prompt = PromptTemplate.from_template(self.refine_template)
        chain = load_summarize_chain(llm, question_prompt=prompt, refine_prompt=refine_prompt, chain_type="refine")
        summary = chain.run(entire_text)
        return summary
    

    def __run_map_reduce_chain(self, entire_text):
        '''
            Creates a Map Reduce LLM Chain for summarizing text
        '''
        list_docs = []
        entire_text = entire_text[0].page_content
        length = int(len(entire_text)/self.map_reduce_divisions)
        for i in range(0, len(entire_text), length):
            list_docs.append(Document(page_content=entire_text[i : min(i+length, len(entire_text))]))
        
        llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))

        map_prompt = PromptTemplate.from_template(self.map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)
        reduce_prompt = PromptTemplate.from_template(self.reduce_template)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="doc_summaries"
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )

        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

        return map_reduce_chain.run(list_docs)

    def summarize(self, entire_text, chain_type):
        '''
            Aggregates functionality to summarize text.
            External functions call this function
        '''

        if(chain_type == "STUFF"):
            return self.__run_stuff_chain(entire_text)

        elif(chain_type == "REFINED"):
            return self.__run_refined_chain(entire_text)  
        
        elif(chain_type == "MAPREDUCE"):
            return self.__run_map_reduce_chain(entire_text)
        
        else:
            return self.__run_stuff_chain(entire_text)



maps = []

def parallel_task(index, content, event):
    prompt = PromptTemplate(template = PARALLEL_MAP_TEMPLATE_SUMMARY, input_variables=["text"])
    llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text=content)
    maps.append((index, summary))
    event.set()

def parallelize_summary(entire_text):
    threads = []
    sentence_per_task = int(os.environ.get("SENTENCE_PER_TASK_PARALLEL_SUMMARY"))
    completion_event = threading.Event()
    tokens = re.split(r'(?<=[.!?])\s+', entire_text)

    for i in range(0, len(tokens), sentence_per_task):
        if i+sentence_per_task > len(tokens):
            t = threading.Thread(target=parallel_task, args=(i//sentence_per_task, tokens[i:], completion_event))
            t.start()
            threads.append(t)
        else:
            t = threading.Thread(target=parallel_task, args=(i//sentence_per_task, tokens[i:i+sentence_per_task], completion_event))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()
    
    sorted_list = sorted(maps, key=lambda x: x[0])
    sum = ""
    for sl in sorted_list:
        sum+=sl[1]
    
    prompt = PromptTemplate(template = PARALLEL_REDUCE_TEMPLATE_SUMMARY, input_variables=["doc_summaries"])
    llm = ChatOpenAI(temperature=0, model_name=os.environ.get("CHAT_MODEL"))
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(doc_summaries=sum)
    return summary