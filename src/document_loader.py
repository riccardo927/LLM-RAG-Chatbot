from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

class DocumentLoader:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_documents(self):
        # Load documents from a directory of PDFs
        loader = PyPDFDirectoryLoader(self.data_path)
        return loader.load()
    
    def split_document(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False
        )
        return text_splitter.split_documents(documents)
