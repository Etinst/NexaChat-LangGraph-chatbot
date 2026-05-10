"""
NexaChat — Enhanced Streamlit Frontend
=========================================
A production-quality chat interface built on top of the LangGraph backend.

Features:
  • Dark glassmorphism UI with smooth CSS animations
  • Multi-thread sidebar with thread switching & creation
  • Persona presets (system-prompt selector)
  • Token usage estimates per message
  • Export conversation as Markdown
  • Streaming responses with typing indicator
  • Tool-use badge shown inline when tools are invoked
  • Keyboard-friendly (Enter to send)

Run:
    streamlit run app.py
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# ── path setup so backend/ and utils/ are importable ──────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ── page config (must be FIRST streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="NexaChat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils import (
    new_thread_id,
    friendly_thread_label,
    truncate,
    estimate_tokens,
    PERSONA_PRESETS,
)

# ──────────────────────────────────────────────
# CSS — glassmorphism dark theme
# ──────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* ── Root & body ────────────────────────────────────────── */
:root {
    --primary:   #6C63FF;
    --primary-light: #9D97FF;
    --bg:        #0F0F1A;
    --surface:   #1A1A2E;
    --surface2:  #16213E;
    --border:    rgba(108,99,255,0.25);
    --text:      #E8E8F0;
    --text-muted:#9090A8;
    --user-bg:   linear-gradient(135deg,#6C63FF,#9D97FF);
    --bot-bg:    rgba(26,26,46,0.85);
    --radius:    14px;
    --shadow:    0 8px 32px rgba(0,0,0,0.45);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: "Inter", "Segoe UI", sans-serif;
}

/* ── Hide default Streamlit elements ────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Inputs ─────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
}

/* ── Chat messages ──────────────────────────────────────── */
.msg-wrapper {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 20px;
    animation: fadeUp 0.35s ease forwards;
}
.msg-wrapper.user  { flex-direction: row-reverse; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0);    }
}

.avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
.avatar.user { background: var(--user-bg); }
.avatar.bot  { background: var(--surface2); border: 1px solid var(--border); }

.bubble {
    max-width: 72%;
    padding: 12px 16px;
    border-radius: var(--radius);
    line-height: 1.6;
    font-size: 0.93rem;
    position: relative;
}
.bubble.user {
    background: var(--user-bg);
    color: #fff;
    border-bottom-right-radius: 4px;
    box-shadow: var(--shadow);
}
.bubble.bot {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(10px);
    box-shadow: var(--shadow);
}
.bubble p { margin: 0 0 6px; }
.bubble p:last-child { margin-bottom: 0; }

.meta {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 4px;
    padding: 0 4px;
}
.msg-wrapper.user .meta { text-align: right; }

/* ── Tool badge ─────────────────────────────────────────── */
.tool-badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 0.7rem;
    background: rgba(108,99,255,0.2);
    color: var(--primary-light);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 20px;
    padding: 2px 8px;
    margin-bottom: 6px;
}

/* ── Typing indicator ───────────────────────────────────── */
.typing-dots {
    display: flex; align-items: center; gap: 5px; padding: 4px 0;
}
.typing-dots span {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--primary);
    animation: bounce 1.2s infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
    40%            { transform: scale(1.0); opacity: 1.0; }
}

/* ── Thread buttons in sidebar ──────────────────────────── */
.thread-btn {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    text-align: left;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    color: var(--text);
    margin-bottom: 8px;
    font-size: 0.85rem;
}
.thread-btn:hover, .thread-btn.active {
    border-color: var(--primary);
    background: rgba(108,99,255,0.15);
}

/* ── Scrollable chat area ───────────────────────────────── */
.chat-scroll {
    height: calc(100vh - 220px);
    overflow-y: auto;
    padding: 12px 4px 24px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}
.chat-scroll::-webkit-scrollbar       { width: 5px; }
.chat-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ── Separator ──────────────────────────────────────────── */
hr.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 16px 0;
}

/* ── Selectbox, buttons ─────────────────────────────────── */
[data-testid="stSelectbox"] > div { background: var(--surface2) !important; border-radius: 8px; }
button[kind="primary"] {
    background: var(--primary) !important;
    border: none !important;
    border-radius: 8px !important;
}
</style>
"""


# ──────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────

def _init_state():
    if "threads" not in st.session_state:
        first_id = new_thread_id()
        st.session_state.threads = {
            first_id: {
                "label": friendly_thread_label(first_id, datetime.now()),
                "created_at": datetime.now(),
                "history": [],   # list of {"role","content","ts","tokens"}
            }
        }
        st.session_state.active_thread = first_id

    if "persona" not in st.session_state:
        st.session_state.persona = "NexaChat (Default)"

    if "streaming" not in st.session_state:
        st.session_state.streaming = True


# ──────────────────────────────────────────────
# Message HTML helpers
# ──────────────────────────────────────────────

def _render_message(role: str, content: str, ts: datetime, tokens: int, used_tool: bool = False):
    avatar = "🧑" if role == "user" else "🤖"
    cls = "user" if role == "user" else "bot"
    time_str = ts.strftime("%-I:%M %p")

    # Sanitise content for HTML (basic)
    import html as _html
    safe_content = _html.escape(content).replace("\n", "<br>")

    tool_badge = (
        '<div class="tool-badge">🔧 Tool used</div>' if used_tool else ""
    )

    bubble_inner = f"{tool_badge}<p>{safe_content}</p>"

    if role == "user":
        html = f"""
        <div class="msg-wrapper user">
          <div class="avatar user">{avatar}</div>
          <div>
            <div class="bubble user">{bubble_inner}</div>
            <div class="meta">{time_str} · ~{tokens} tokens</div>
          </div>
        </div>"""
    else:
        html = f"""
        <div class="msg-wrapper bot">
          <div class="avatar bot">{avatar}</div>
          <div>
            <div class="bubble bot">{bubble_inner}</div>
            <div class="meta">{time_str} · ~{tokens} tokens</div>
          </div>
        </div>"""

    st.markdown(html, unsafe_allow_html=True)


def _typing_indicator():
    st.markdown(
        """<div class="msg-wrapper bot">
          <div class="avatar bot">🤖</div>
          <div class="bubble bot">
            <div class="typing-dots">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown("## 💬 NexaChat")
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── New conversation button ────────────────────────
        if st.button("＋  New Conversation", use_container_width=True, type="primary"):
            tid = new_thread_id()
            st.session_state.threads[tid] = {
                "label": friendly_thread_label(tid, datetime.now()),
                "created_at": datetime.now(),
                "history": [],
            }
            st.session_state.active_thread = tid
            st.rerun()

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Thread list ────────────────────────────────────
        st.markdown("**Conversations**")
        for tid, meta in reversed(list(st.session_state.threads.items())):
            is_active = tid == st.session_state.active_thread
            label = meta["label"]
            history = meta["history"]
            preview = truncate(history[-1]["content"], 45) if history else "Empty chat"
            btn_label = f"{'▶ ' if is_active else ''}{label}\n{preview}"
            if st.button(btn_label, key=f"thread_{tid}", use_container_width=True):
                st.session_state.active_thread = tid
                st.rerun()

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Persona selector ───────────────────────────────
        st.markdown("**Assistant Persona**")
        persona = st.selectbox(
            label="persona",
            options=list(PERSONA_PRESETS.keys()),
            index=list(PERSONA_PRESETS.keys()).index(st.session_state.persona),
            label_visibility="collapsed",
        )
        if persona != st.session_state.persona:
            st.session_state.persona = persona
            st.rerun()

        persona_desc = {
            "NexaChat (Default)": "General-purpose assistant with tool access.",
            "Code Mentor":        "Senior engineer who explains the 'why'.",
            "Writing Coach":      "Journalism-trained editor for clearer prose.",
            "Data Analyst":       "Python data expert (pandas, numpy, etc.).",
            "Explain Like I'm 5": "Complex ideas in simple, fun language.",
        }
        st.caption(persona_desc.get(persona, ""))

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Settings ───────────────────────────────────────
        st.markdown("**Settings**")
        st.session_state.streaming = st.toggle(
            "Streaming responses", value=st.session_state.streaming
        )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Export ─────────────────────────────────────────
        active_history = st.session_state.threads[st.session_state.active_thread]["history"]
        if active_history:
            md_lines = [f"# NexaChat Export\n*Persona: {st.session_state.persona}*\n\n---\n"]
            for msg in active_history:
                role_label = "**You**" if msg["role"] == "user" else "**NexaChat**"
                md_lines.append(f"{role_label}: {msg['content']}\n")
            md_export = "\n".join(md_lines)
            st.download_button(
                "📥 Export Conversation",
                data=md_export,
                file_name="nexachat_export.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # ── Clear current thread ───────────────────────────
        if st.button("🗑 Clear This Chat", use_container_width=True):
            st.session_state.threads[st.session_state.active_thread]["history"] = []
            # Replace thread in graph memory by creating a new thread id
            new_tid = new_thread_id()
            meta = st.session_state.threads.pop(st.session_state.active_thread)
            meta["history"] = []
            st.session_state.threads[new_tid] = meta
            st.session_state.active_thread = new_tid
            st.rerun()

        # ── Footer ─────────────────────────────────────────
        st.markdown(
            "<br><div style='color:#9090A8;font-size:0.75rem;text-align:center'>"
            "Built with LangGraph · Groq · Streamlit</div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# Main chat area
# ──────────────────────────────────────────────

def _chat_area():
    thread = st.session_state.threads[st.session_state.active_thread]
    history = thread["history"]

    st.markdown(
        f"<h2 style='margin:0;font-size:1.4rem;color:#E8E8F0;'>"
        f"💬 {thread['label']}"
        f"</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Render history ─────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        if not history:
            st.markdown(
                """<div style="text-align:center;margin-top:80px;color:#9090A8;">
                    <div style="font-size:3rem;margin-bottom:12px;">🤖</div>
                    <div style="font-size:1.1rem;font-weight:600;margin-bottom:8px;">Hello! I'm NexaChat</div>
                    <div style="font-size:0.9rem;">
                      Ask me anything — I can do maths, check the time, analyse text, and more.
                    </div>
                  </div>""",
                unsafe_allow_html=True,
            )
        else:
            for msg in history:
                _render_message(
                    role=msg["role"],
                    content=msg["content"],
                    ts=msg["ts"],
                    tokens=msg["tokens"],
                    used_tool=msg.get("tool_used", False),
                )

    # ── Input ──────────────────────────────────────────────
    user_input = st.chat_input("Message NexaChat…")
    if not user_input:
        return

    # Save user message
    user_tokens = estimate_tokens(user_input)
    user_msg = {
        "role": "user",
        "content": user_input,
        "ts": datetime.now(),
        "tokens": user_tokens,
        "tool_used": False,
    }
    history.append(user_msg)
    _render_message("user", user_input, user_msg["ts"], user_tokens)

    # ── Call backend ───────────────────────────────────────
    try:
        from backend.graph import chat, stream_chat

        system_prompt = PERSONA_PRESETS[st.session_state.persona]
        tid = st.session_state.active_thread

        if st.session_state.streaming:
            # Streaming: write tokens into a placeholder
            placeholder = st.empty()
            full_response = ""
            with placeholder:
                _typing_indicator()

            for chunk in stream_chat(user_input, tid, system_prompt=system_prompt):
                full_response += chunk
                with placeholder:
                    _render_message(
                        "assistant",
                        full_response + "▌",
                        datetime.now(),
                        estimate_tokens(full_response),
                    )

            placeholder.empty()
            reply = full_response
        else:
            with st.spinner("NexaChat is thinking…"):
                reply = chat(user_input, tid, system_prompt=system_prompt)

        # detect if a tool was used (crude: look for "Result:" prefix)
        tool_used = reply.strip().startswith("Result:") or "[tool]" in reply.lower()

        bot_msg = {
            "role": "assistant",
            "content": reply,
            "ts": datetime.now(),
            "tokens": estimate_tokens(reply),
            "tool_used": tool_used,
        }
        history.append(bot_msg)
        _render_message("assistant", reply, bot_msg["ts"], bot_msg["tokens"], tool_used)

    except EnvironmentError as e:
        st.error(f"⚠️ Configuration error: {e}")
        st.info(
            "Make sure you have a `.env` file with `GROQ_API_KEY=your_key_here`. "
            "Get a free key at https://console.groq.com"
        )
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    _init_state()
    _sidebar()
    _chat_area()


if __name__ == "__main__":
    main()
