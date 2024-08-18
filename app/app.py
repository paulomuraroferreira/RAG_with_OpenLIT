from fastapi import FastAPI
from pydantic import BaseModel
from retriver import main

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
def query_rag(query: Query):
    print(query)
    response = main(query.question)
    return {"answer": response['answer']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
