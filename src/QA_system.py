from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question: {question}
"""

class QuestionAnsweringSystem:
    def __init__(self, document_embedder, chroma_db_path):
        self.document_embedder = document_embedder
        self.db = self.document_embedder.load_chroma(chroma_db_path)
    
    def ask_question(self, query):
        # Search the DB.
        results = self.db.similarity_search_with_score(query, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=query)

        # Model
        # model = Ollama(model="mistral")
        # model = Ollama(model="llama3:latest")
        model = Ollama(model="llama3.1:8b")

        response_text = model.invoke(prompt)
        sources = [doc.metadata.get("id", None) for doc, _score in results]

        formatted_response = f"Response: {response_text}\nSources: {sources}"
        # print(formatted_response)

        # formatted_source = f"nSources: {sources}"
        # print(formatted_source)
        return formatted_response

