"""
Defines the final loading pipeline from Data loading to getting chunks
"""
# UNCOMMENT 3 LINES
from R4B.FileLoaders.loader import Loader
from R4B.FileLoaders.preprocess import Preprocess
from R4B.FileLoaders.chunk import Chunk

class DataPipeline:
    """
    Defines the entire pipeline depending on the extension
    Allowed extensions are defined in the Options dictionary
        - pdf
        - txt
        - docx
    """

    def __init__(self, chunk_list=["Entire Document", "Sentence", "Tokens", "Title"]):
        '''
            1. extract extension from the document path.
            2. according to the extension call the appropriate document loader.
            3. according to extension call the appropriate data loader.
            4. call the chunking algorithm according to the extension and desire.
        '''
        self.loader = Loader()
        self.Loading_Options = {
            "pdf": self.loader.read_html_pdf,
            "txt": self.loader.read_naive_txt,
            "docx":self.loader.read_naive_docx
        }

        self.options = {
            "pdf": self.preprocess_pdf,
            "txt": self.preprocess_txt,
            "docx":self.preprocess_docx
        }

        self.chunk_list = chunk_list
        self.preprocess = Preprocess()
        self.chunk = Chunk()
    
    def preprocess_pdf(self, document_path, chunk_size, chunk_overlap):
        data = self.loader.read_html_pdf(document_path)
        preprocessed_content = self.preprocess.preprocess("HTMLParse", data)

        chunk_dict = {}
        for chunk_option in self.chunk_list:
            chunk_dict[chunk_option] = []
        
        for i in self.chunk_list:
            if i=="Tokens":
                for d in [preprocessed_content["entire_doc"]]:
                    chunk_dict[i]+=self.chunk.chunk(chunk_size, chunk_overlap, d, "NaiveTokenChunk")
            if i=="Title":
                chunk_dict[i] = self.chunk.chunk(chunk_size, chunk_overlap, preprocessed_content["title"], "HTMLBS4Chunk")
            if i=="Entire Document":
                chunk_dict[i] = self.chunk.chunk(chunk_size, chunk_overlap, preprocessed_content["entire_doc"], "EntireDoc")
            if i=="Sentence":
                chunk_dict[i] = self.chunk.chunk(chunk_size, chunk_overlap, preprocessed_content["entire_doc"], "SentenceChunk")
        return chunk_dict
    
    
    def preprocess_txt(self, document_path, chunk_size, chunk_overlap):
        data = self.loader.read_naive_txt(document_path)
        preprocessed_content = self.preprocess.preprocess("NaiveText", data)

        chunk_dict = {}
        for chunk_option in self.chunk_list:
            chunk_dict[chunk_option] = []

        
        for i in self.chunk_list:
            if i=="Tokens":
                for d in [preprocessed_content["entire_doc"]]:
                    chunk_dict[i]+=self.chunk.chunk(chunk_size, chunk_overlap, d, "NaiveTokenChunk")
            if i=="Title":
                chunk_dict[i] = []
            if i=="Entire Document":
                chunk_dict[i] = self.chunk.chunk(chunk_size, chunk_overlap, preprocessed_content["entire_doc"], "EntireDoc")
            if i=="Sentence":
                chunk_dict[i] = self.chunk.chunk(chunk_size, chunk_overlap, preprocessed_content["entire_doc"], "SentenceChunk")
        return chunk_dict
        

    def preprocess_docx(self, document_path, chunk_size, chunk_overlap):
        raise RuntimeError("Please convert your docx file to pdf before uploading <3.")


    def process_document(self, document_path, chunk_size, chunk_overlap):
        print("Processing document: " + document_path)
        extension = document_path.split(".")[-1]
        if extension in self.options:
            return self.options[extension](document_path, chunk_size, chunk_overlap)
        else:
            raise RuntimeError("Invalid File Extension. We only accept pdf or txt files.")
        


        # for d in preprocessed_content:
        #     chunks+=self.chunk.chunk(chunk_size, chunk_overlap, d, "HTMLBS4Chunk")
        # return {"source": data, "chunks": chunks, "metadata": {"path": document_path, "chunk_type": "NaiveTokenChunk",
        #                                                        "preprocess_type": "NaiveText"}}




#pipeline tests
# if __name__ == '__main__':
#     p = DataPipeline()
#     pp = p.process_document("FileLoaders/p.pdf", 384, 20)
#     print(len(pp["Title"]))
#     print(len(pp["Sentence"]))
#     print(len(pp["Tokens"]))