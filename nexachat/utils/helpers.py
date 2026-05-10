"""
NexaChat - Thread & Session Utilities
======================================
Helpers for managing conversation threads, session state,
and UI formatting utilities used by the Streamlit frontend.
"""

import uuid
from datetime import datetime


# ──────────────────────────────────────────────
# Thread Management
# ──────────────────────────────────────────────

def new_thread_id() -> str:
    """Generate a universally unique thread ID."""
    return str(uuid.uuid4())


def friendly_thread_label(thread_id: str, created_at: datetime | None = None) -> str:
    """
    Convert a UUID thread ID to a short human-readable label.

    Example: 'Chat · 3a1f' or 'Chat · 3a1f · Jun 12 14:05'
    """
    short = thread_id[:4].upper()
    if created_at:
        ts = created_at.strftime("%b %-d %H:%M")
        return f"Chat · {short} · {ts}"
    return f"Chat · {short}"


# ──────────────────────────────────────────────
# Message Formatting
# ──────────────────────────────────────────────

def format_message_time(dt: datetime) -> str:
    """Return a short human-readable time string like '2:34 PM'."""
    return dt.strftime("%-I:%M %p")


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis for sidebar previews."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (1 token ≈ 4 chars).
    Used purely for display purposes in the UI.
    """
    return max(1, len(text) // 4)


# ──────────────────────────────────────────────
# Persona / System Prompt Presets
# ──────────────────────────────────────────────

PERSONA_PRESETS = {
    "NexaChat (Default)": (
        "You are NexaChat, a helpful, concise, and friendly AI assistant. "
        "You have access to tools: a calculator, a datetime checker, and a word counter. "
        "Use them when the user asks for calculations, time, or text statistics. "
        "Always be transparent when you use a tool."
    ),
    "Code Mentor": (
        "You are a senior software engineer and patient mentor. "
        "When answering coding questions, always explain the 'why' behind your solutions, "
        "suggest best practices, and mention potential pitfalls. "
        "Prefer Python examples unless otherwise specified."
    ),
    "Writing Coach": (
        "You are a professional writing coach with a background in journalism. "
        "Help users improve clarity, structure, and tone in their writing. "
        "Be encouraging but honest. Point out weak sentences and suggest rewrites."
    ),
    "Data Analyst": (
        "You are a data analyst specializing in Python (pandas, numpy, matplotlib). "
        "Help users understand their data, suggest analyses, write clean data-processing code, "
        "and interpret results. Always ask about the dataset shape and goal before diving in."
    ),
    "Explain Like I'm 5": (
        "You explain complex topics using simple words, fun analogies, and short sentences. "
        "Imagine you are talking to a bright 10-year-old who is curious about everything. "
        "Avoid jargon. If you must use a technical term, immediately define it simply."
    ),
}
