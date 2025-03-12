import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Initialize Ollama model
try:
    llm = OllamaLLM(model="dolphin3:latest")
    logging.info("✅ Ollama model loaded successfully")
except Exception as e:
    logging.error(f"❌ Failed to load Ollama model: {e}")
    raise HTTPException(status_code=500, detail="Error loading LLM")

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
        logging.info(f"Received message: {request.message}")

        user_input = request.message
        formatted_prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nEchoBot:"

        # Generate response
        response = llm.invoke(formatted_prompt)

        logging.info(f"Generated response: {response}")
        return {"response": response}

    except Exception as e:
        logging.error(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
