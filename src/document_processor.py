from document_loader import DocumentLoader
from chroma_handler import ChromaHandler

class DocumentProcessor:
    def __init__(self, data_path, database_path, embedder):
        self.data_path = data_path
        self.database_path = database_path
        self.embedder = embedder
        self.loader = DocumentLoader(self.data_path)  # Use the updated loader

    def reset_database(self):
        chroma_handler = ChromaHandler(self.database_path, self.embedder.embedding_type)
        chroma_handler.clear_database()

    def process_documents(self):
        # Load and split documents
        documents = self.loader.load_documents()
        chunks = self.loader.split_document(documents)

        # Add chunks to Chroma
        chroma_handler = ChromaHandler(self.database_path, self.embedder.embedding_type)
        chroma_handler.add_to_chroma(chunks)
        print(f"Processed and added {len(chunks)} chunks to the Chroma database.")