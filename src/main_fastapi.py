import logging
from fastapi import FastAPI
from pydantic import BaseModel
from document_embedder import DocumentEmbedder
from document_processor import DocumentProcessor
from QA_system import QuestionAnsweringSystem
import uvicorn

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = '/Users/riccardoelhassanin/Desktop/ChatBot/data'
DATABASE_PATH = '/Users/riccardoelhassanin/Desktop/ChatBot/data/database'

# Embedding and Credentials Info
EMBEDDING_TYPE = "sentence-transformers"  # Change this based on the model you are using
CREDENTIALS = None
REGION_NAME = None
API_KEY = None

# Initialize the embedder, document processor, and question answering system
embedder = DocumentEmbedder(embedding_type=EMBEDDING_TYPE, credentials=CREDENTIALS, region_name=REGION_NAME, api_key=API_KEY)
doc_processor = DocumentProcessor(DATA_PATH, DATABASE_PATH, embedder)
qa_system = QuestionAnsweringSystem(embedder, DATABASE_PATH)

# Initialize FastAPI app
app = FastAPI()

# Pydantic model to parse the query request
class QueryRequest(BaseModel):
    question: str

@app.post("/ask_question")
async def ask_question(request: QueryRequest):
    query = request.question
    try:
        # Get the response from the QA system
        response = qa_system.ask_question(query)
        logger.info(f"Query: {query}, Response: {response}")
        return {"query": query, "response": response}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}

@app.post("/reset")
async def reset_database():
    try:
        logger.info("✨ Clearing Database")
        doc_processor.reset_database()
        return {"message": "Database cleared"}
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        return {"error": str(e)}

@app.post("/process")
async def process_documents():
    try:
        logger.info("📄 Processing and embedding documents")
        doc_processor.process_documents()
        return {"message": "Documents processed and embedded"}
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return {"error": str(e)}

# Run the FastAPI server using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# @app.post("/ask_question/")
# async def ask_question(query: str):
#     try:
#         # Process the question locally using the embeddings and QA system
#         answer = qa_system.ask_question(query)
#         return {"answer": answer}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     # Run the FastAPI server on port 8000
#     uvicorn.run(app, host="0.0.0.0", port=8000)



