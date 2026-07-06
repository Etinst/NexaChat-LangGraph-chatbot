from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

APP_TITLE = "NexaChat Groq"
DEFAULT_THREAD_ID = "darshan-thread-1"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DATABASE_PATH = Path(os.getenv("NEXACHAT_DB", "nexachat_threads.db"))


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Add it to your .env file before running NexaChat."
        )
    return value


def build_groq_llm(model: str | None = None, temperature: float = 0.2) -> ChatGroq:
    require_env("GROQ_API_KEY")
    return ChatGroq(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        max_retries=2,
    )


def get_alpha_vantage_key() -> str | None:
    return os.getenv("ALPHAVANTAGE_API_KEY")
