from fastapi import FastAPI
from pydantic import BaseModel
from query import ask  # assuming ask() takes a string and returns a string

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_endpoint(data: AskRequest):
    answer = ask(data.question)
    return {"response": answer}

