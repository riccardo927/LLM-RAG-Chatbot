import crewai
from document_embedder import DocumentEmbedder
from QA_system import QuestionAnsweringSystem

DATABASE_PATH = '/Users/riccardoelhassanin/Desktop/ChatBot/data/database'

# Initialize CrewAI worker
worker = crewai.Worker(name="question_answer_worker")

@worker.task()
def answer_question(query: str, embedding_type: str, model_name: str = None):
    # Initialize embedder and QA system
    embedder = DocumentEmbedder(embedding_type=embedding_type, model_name=model_name)
    qa_system = QuestionAnsweringSystem(embedder, DATABASE_PATH)
    
    # Answer the query
    answer = qa_system.ask_question(query)
    return {"answer": answer}

if __name__ == "__main__":
    worker.run()
