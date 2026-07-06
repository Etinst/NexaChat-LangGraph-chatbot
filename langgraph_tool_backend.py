from __future__ import annotations

import sqlite3
from typing import Annotated, Literal, TypedDict

import requests
from app_settings import DATABASE_PATH, build_groq_llm, get_alpha_vantage_key
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

llm = build_groq_llm()
web_search = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_number: float,
    second_number: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> dict:
    """Do one basic math operation and return the result."""
    if operation == "add":
        result = first_number + second_number
    elif operation == "subtract":
        result = first_number - second_number
    elif operation == "multiply":
        result = first_number * second_number
    elif operation == "divide":
        if second_number == 0:
            return {"error": "Cannot divide by zero."}
        result = first_number / second_number
    else:
        return {"error": "Unsupported operation."}
    return {"operation": operation, "result": result}


@tool
def stock_quote(symbol: str) -> dict:
    """Fetch a recent stock quote using Alpha Vantage when an API key is configured."""
    symbol = symbol.strip().upper()
    if not symbol:
        return {"error": "Please provide a stock symbol, for example AAPL or MSFT."}

    api_key = get_alpha_vantage_key()
    if not api_key:
        return {"error": "Add ALPHAVANTAGE_API_KEY to .env to use stock quotes."}

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return {"error": f"Stock quote request failed: {exc}"}

    if not data.get("Global Quote"):
        return {"error": f"No stock quote found for {symbol}.", "raw_response": data}
    return data


tools = [web_search, calculator, stock_quote]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def tool_chat_node(state: ChatState) -> dict:
    system_message = SystemMessage(
        content=(
            "You are NexaChat with tools. Use search for current facts, "
            "calculator for math, and stock_quote for market quotes. "
            "When a tool is not needed, answer directly."
        )
    )
    response = llm_with_tools.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
checkpointer = SqliteSaver(connection)

graph = StateGraph(ChatState)
graph.add_node("tool_chat_node", tool_chat_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "tool_chat_node")
graph.add_conditional_edges("tool_chat_node", tools_condition)
graph.add_edge("tools", "tool_chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads() -> list[str]:
    thread_ids: set[str] = set()
    for checkpoint in checkpointer.list(None):
        thread_ids.add(checkpoint.config["configurable"]["thread_id"])
    return sorted(thread_ids)
