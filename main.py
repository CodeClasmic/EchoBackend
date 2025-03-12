import os
import logging
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Get OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logging.error("❌ OPENAI_API_KEY is missing! Set it in your .env file.")
    raise RuntimeError("Missing OpenAI API Key!")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

# System Prompt
SYSTEM_PROMPT = """You are an AI assistant named EchoBot. Your purpose is to assist users by answering questions, providing helpful information, and engaging in conversations. Keep responses professional and useful."""

# Define request model
class ChatRequest(BaseModel):
    message: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to EchoBot Backend (OpenAI)!"}

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logging.info(f"📩 Received message: {request.message}")

        # OpenAI API Call
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Change to "gpt-3.5-turbo" if needed
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ],
            temperature=0.7,
        )

        bot_response = response["choices"][0]["message"]["content"]
        logging.info(f"✅ Generated response: {bot_response}")

        return {"response": bot_response}

    except openai.error.OpenAIError as e:
        logging.error(f"❌ OpenAI API Error: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API Error")

    except Exception as e:
        logging.error(f"❌ Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
