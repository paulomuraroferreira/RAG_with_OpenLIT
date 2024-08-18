from fastapi import FastAPI
from pydantic import BaseModel
from retriver import main

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
def query_rag(query: Query):
    response = main(query.question)
    return {"answer": response['answer'],
            "context": response['context'],
            "question": response['question']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
