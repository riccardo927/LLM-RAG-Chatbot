import logging
from document_embedder import DocumentEmbedder
from document_processor import DocumentProcessor
from QA_system import QuestionAnsweringSystem

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the embedder and question answering system
DATA_PATH = '/Users/riccardoelhassanin/Desktop/ChatBot/data'
DATABASE_PATH = '/Users/riccardoelhassanin/Desktop/ChatBot/data/database'

EMBEDDING_TYPE = "sentence-transformers"  # Change this based on the model you are using
CREDENTIALS = None
REGION_NAME = None
API_KEY = None

def main(f):
    if f == 'reset':
        logger.info("✨ Clearing Database")
        doc_processor.reset_database()
    
    logger.info("📄 Processing and embedding documents")
    doc_processor.process_documents()

def chat():
    logger.info("Welcome to the Chatbot! Type 'exit' to stop.")
    
    while True:
        # Get the user input (question)
        query = input("You: ")

        if query.lower() == "exit":
            logger.info("Goodbye!")
            break

        try:
            # Get the response from the QA system
            response = qa_system.ask_question(query)
            logger.info(f"Chatbot: {response}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    # Initialize the embedder, document processor, and question answering system
    embedder = DocumentEmbedder(embedding_type=EMBEDDING_TYPE, credentials=CREDENTIALS, region_name=REGION_NAME, api_key=API_KEY)
    doc_processor = DocumentProcessor(DATA_PATH, DATABASE_PATH, embedder)
    
    f = input("Enter 'reset' to clear the database, or any key to process documents: ")
    main(f)
    
    qa_system = QuestionAnsweringSystem(embedder, DATABASE_PATH)
    chat()
