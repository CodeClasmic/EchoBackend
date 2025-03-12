import os
import logging
import requests
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

# Ollama Server URL
OLLAMA_HOST = "http://127.0.0.1:11434"

# Check if Ollama is running
def is_ollama_running():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if response.status_code == 200:
            return True
        return False
    except requests.exceptions.RequestException:
        return False

# Initialize Ollama model only if server is running
if is_ollama_running():
    try:
        llm = OllamaLLM(model="dolphin3:latest")
        logging.info("✅ Ollama model loaded successfully")
    except Exception as e:
        logging.error(f"❌ Failed to load Ollama model: {e}")
        llm = None
else:
    logging.error("❌ Ollama is not running! Start Ollama with 'ollama serve'.")
    llm = None

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
    if llm is None:
        logging.error("❌ Ollama model is not initialized")
        raise HTTPException(status_code=503, detail="Ollama service unavailable")

    try:
        logging.info(f"📩 Received message: {request.message}")

        user_input = request.message
        formatted_prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nEchoBot:"

        # Generate response
        response = llm.invoke(formatted_prompt)

        logging.info(f"✅ Generated response: {response}")
        return {"response": response}

    except Exception as e:
        logging.error(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
