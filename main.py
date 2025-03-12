import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()

# Environment Variables
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Enable LangChain tracing
os.environ["LANGCHAIN_TRACING_v2"] = "true"

# Initialize FastAPI app
app = FastAPI()

# Initialize Ollama model
llm = OllamaLLM(model="dolphin3:latest")

# System Prompt
SYSTEM_PROMPT = """You are an AI assistant named EchoBot. Your purpose is to assist users by answering questions, providing helpful information, and engaging in conversations. Keep responses professional and useful."""

# Define request model
class ChatRequest(BaseModel):
    message: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to EchoBot Backend!"}

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_input = request.message
        formatted_prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nEchoBot:"

        # Generate response
        response = llm.invoke(formatted_prompt)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
