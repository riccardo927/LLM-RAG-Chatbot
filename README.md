# LLM-RAG-Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot using Large Language Models (LLMs) to answer questions based on specific document embeddings. It leverages Flask for deployment, document processing with Python, and document similarity search with Chroma.

## Features

- **Modular Design**: Follows Object-Oriented Programming (OOP) principles to ensure flexibility and scalability.
- **Document Embedding**: Uses embeddings to represent document content for efficient similarity search.
- **Question Answering**: Responds to user queries by retrieving relevant documents and generating context-aware answers.
- **Chroma for Vector Search**: Handles the document search and retrieval process.
- **Flask**: Provides an API interface for the chatbot. (FastAPI is under development)

## Setup and Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.10+
- pip
- `virtualenv` or `venv` (recommended for creating isolated environments)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/riccardo927/LLM-RAG-Chatbot.git
    cd LLM-RAG-Chatbot
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the chatbot locally

* Run the `main.py` script:

    ```bash
    cd src
    python main.py
    ```

### Running the Flask Application

1. Start the Flask application:

    ```bash
    cd src
    python main_flask.py
    ```

2. Navigate to the provided URL in your browser (`http://127.0.0.1:5001`) to interact with the chatbot.


** The FastAPI application has not yet been implemented fully. 

