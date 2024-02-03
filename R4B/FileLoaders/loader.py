"""
Service for loading text files
"""

from langchain.document_loaders import UnstructuredPDFLoader,  TextLoader, PDFMinerPDFasHTMLLoader
from langchain.docstore.document import Document
# import docx
# import mammoth
from langchain.document_loaders import PDFMinerPDFasHTMLLoader



class Loader:
    def __init__(self):
        pass

    def read_naive_pdf(self, path_to_file):
        '''
            Takes as input path to a pdf file
            Reads using an unstructured PDF loader
            Returns the list of contents wrapped in a langchain Document
        '''
        loader = UnstructuredPDFLoader(path_to_file)
        data = loader.load()
        return data


    def read_naive_txt(self, path_to_file):
        '''
            Takes as input, path to a txt file
            Reads using langchain's TextLoader
            Returns the list of contents.
        ''' 
        loader = TextLoader(path_to_file)
        data = loader.load()
        return data
    

    def read_naive_docx(self, path_to_file):
        '''
            Takes as input, path to a docx file
            Reads using docx
            Returns list of contents
        # '''
        # doc = docx.Document(path_to_file)
        # content = ""
        # for i in doc.paragraphs:
        #     content += i.text
        
        return [Document(page_content="content", metadata={})]
    

    def read_html_pdf(self, path_to_file):
        '''
            Takes as input, path to a pdf file
            Reads data using langchains PDFLoader
            Returns an html version of the PDF
        '''
        loader = PDFMinerPDFasHTMLLoader(path_to_file)
        data = loader.load()
        return data


    def read_html_docx(self, path_to_file):
        '''
            Takes as input, path to a docx file
            Reads data using mammoth
            Returns an html version of the docx file
        '''
        # with open(path_to_file, "rb") as docx_file:
        #     result = mammoth.convert_to_html(docx_file)
        # return [Document(page_content=result.value, metadata={})]
        return [Document(page_content="content", metadata={})]
