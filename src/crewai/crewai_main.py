from fastapi import FastAPI
import crewai

app = FastAPI()

# Initialize the CrewAI client
APY_KEY = "crewai_api_key"
crewai_client = crewai.Client(api_key=APY_KEY)

@app.post("/ask_question/")
async def ask_question(query: str, embedding_type: str):
    # Trigger CrewAI job
    job_id = crewai_client.jobs.create(
        task="answer_question",
        arguments={"query": query, "embedding_type": embedding_type}
    )
    
    # Optionally, wait for the result
    result = crewai_client.jobs.get_result(job_id)
    
    return result
