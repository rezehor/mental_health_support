import os
import sys

from openai import OpenAI
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=1000)

@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    try:
        system_message = {
            "role": "system",
            "content": (
                "Ти — добрий і підтримуючий психологічний помічник. "
                "Ти слухаєш уважно, відповідаєш емпатично, ввічливо і з розумінням. "
                "Твої відповіді завжди підтримують, допомагають справитися з тривогою, стресом і труднощами. "
                "Говори українською."
            )
        }
        completion = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[
                system_message,
                {"role": "user", "content": request.message}
            ],
            temperature=0.6,
            max_tokens=512,
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "local-dev",
            }
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
