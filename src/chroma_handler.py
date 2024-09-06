import os
import shutil
from langchain_chroma.vectorstores import Chroma
from langchain.schema import Document
from document_embedder import DocumentEmbedder

class ChromaHandler:
    def __init__(self, database_path, embedding_type, credentials=None, region_name=None, api_key=None):
        self.database_path = database_path
        self.embedder = DocumentEmbedder(embedding_type, credentials, region_name, api_key)


    def calculate_chunk_ids(self, chunks):

        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks

    def add_to_chroma(self, chunks: list[Document]):
        db = self.embedder.load_chroma(self.database_path)

        # Calculate chunk IDs
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Get existing document IDs in the database
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Filter out new chunks that are not already in the database
        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("✅ No new documents to add")

    def clear_database(self):
        if os.path.exists(self.database_path):
            shutil.rmtree(self.database_path)
        # os.makedirs(self.database_path, exist_ok=True)
