import logging
from flask import Flask, render_template, request
from document_embedder import DocumentEmbedder
from document_processor import DocumentProcessor
from QA_system import QuestionAnsweringSystem

# Initialize Flask app
app = Flask(__name__)

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths for data and database
DATA_PATH = '/Users/riccardoelhassanin/Desktop/ChatBot/data'
DATABASE_PATH = '/Users/riccardoelhassanin/Desktop/ChatBot/data/database'

# Embedding and credentials info
EMBEDDING_TYPE = "sentence-transformers"  # Adjust based on the model you're using
CREDENTIALS = None
REGION_NAME = None
API_KEY = None

# Initialize the embedder, document processor, and question answering system
embedder = DocumentEmbedder(embedding_type=EMBEDDING_TYPE, credentials=CREDENTIALS, region_name=REGION_NAME, api_key=API_KEY)
doc_processor = DocumentProcessor(DATA_PATH, DATABASE_PATH, embedder)
qa_system = QuestionAnsweringSystem(embedder, DATABASE_PATH)

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def ask_question():
    question = request.form.get('question', '')
    
    if not question:
        return render_template('index.html', question="", answer="Please enter a question.")
    
    try:
        # Get the response from the QA system
        logger.info(f"Received question: {question}")
        answer = qa_system.ask_question(question)
        logger.info(f"Generated answer: {answer}")
        return render_template('index.html', question=question, answer=answer)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return render_template('index.html', question=question, answer="Error processing question")

@app.route('/reset_database', methods=['POST'])
def reset_database():
    try:
        logger.info("✨ Clearing Database")
        doc_processor.reset_database()
        return render_template('index.html', question="", answer="", message="Database reset successful!")
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        return render_template('index.html', question="", answer="", message=f"Error resetting database: {str(e)}")

@app.route('/process_documents', methods=['POST'])
def process_documents():
    try:
        logger.info("📄 Processing and embedding documents")
        doc_processor.process_documents()
        return render_template('index.html', question="", answer="", message="Documents processed and embedded successfully!")
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return render_template('index.html', question="", answer="", message=f"Error processing documents: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
