import os
import sys
from enum import Enum
from typing import Annotated, List
from openai import OpenAI
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS"),
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


class RoleEnum(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: RoleEnum
    content: str


class ChatRequest(BaseModel):
    messages: Annotated[List[ChatMessage], Field(min_items=1)]


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)

async def chat(request: ChatRequest) -> ChatResponse:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[message.model_dump() for message in request.messages],
            temperature=0.6,
            max_tokens=512,
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "local-dev",
            }
        )
        content = completion.choices[0].message.content
        return ChatResponse(response=content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
