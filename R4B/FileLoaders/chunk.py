'''
Service to chunk data based on various functionalities
'''

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import re
import uuid

tiktoken_encoder = tiktoken.get_encoding("gpt2")

class Chunk:
    """
    Defines the different chunking algorithms
    Options involve
        - NaiveCharacterChunk
        - NaiveTokenChunk
        - PDFBS4Chunk
    """

    def __init__(self, add_title_to_chunk=True):
        self.Options = {
            "NaiveCharacterChunk": self.naive_recursive_chunk_character,
            "NaiveTokenChunk": self.naive_recursive_chunk_token,
            "HTMLBS4Chunk": self.html_bs4_chunk,
            "SentenceChunk": self.sentence_chunk,
            "EntireDoc": self.entire_doc
        }
        self.add_title_to_chunk = True


    def chunk(self, chunk_size, chunk_overlap, content, chunking_algorithm):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if chunking_algorithm in self.Options:
            return self.Options[chunking_algorithm](content)
        else:
            raise RuntimeError("Invalid chunking algorithm")

    def naive_recursive_chunk_character(self, content):
        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("Chunk overlap cannot be greater than or equal to chunk size")
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap  = self.chunk_overlap, length_function = len, is_separator_regex = False)
        texts = text_splitter.split_text(content)
        for text in texts:
            documents.append(Document(page_content=text, metadata={}))
        return documents


    def naive_recursive_chunk_token(self, content):
        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("Chunk overlap cannot be greater than or equal to chunk size")
        documents = []
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_text(content)
        for text in texts:
            documents.append(Document(page_content=text, metadata={}))
        return documents


    def html_bs4_chunk(self, list_of_title_chunks):
        documents = []
        for chunk in list_of_title_chunks:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            texts = text_splitter.split_text(chunk["content"])
            title = chunk["title"]
            pages = chunk["pages"]
            for text in texts:
                documents.append(Document(page_content=title+"\n"+text, metadata={"title": title, "pages": pages}))
        return documents
    

    def entire_doc(self, content):
        return Document(page_content=content, metadata={})


    def sentence_chunk(self, content):
        # self.chunk_size defines when to stop combining sentences.

        tokens = re.split(r'(?<=[.!?])\s+', content)
        documents = []
        s = ""

        for i in range(len(tokens)):
            if len(s.split())<self.chunk_size:
                s+=" "+tokens[i]
            else:
                documents.append(Document(page_content=s, metadata={}))
                s = tokens[i]
            if (i==len(tokens)-1):
                documents.append(Document(page_content=s, metadata={}))
        return documents