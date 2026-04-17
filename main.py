from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict, Optional

from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from PIL import Image
import io

# ---------- Load API key ----------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

app = FastAPI()

# Simple toggle if you ever want to disable debug logs
DEBUG_MODE = True

# ---------- Enable CORS for frontend ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Input model ----------
class ChatRequest(BaseModel):
    message: str
    session_id: str

# ---------- Define tools ----------
# Wikipedia Tool
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=200
    )
)

# Python REPL Tool
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description=(
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out with print(...)."
    ),
    func=python_repl.run,
)

# ---------- LLMs ----------
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant",  # or any other valid Groq model
    temperature=0
)

llm_with_tools = llm.bind_tools(tools=[wiki_tool, repl_tool])

# ---------- Graph state ----------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: Optional[str]   # "tools" or "general"
    agent: Optional[str]   # "Planner", "Tool Agent", "General Agent"

# ---------- LangGraph definition ----------
graph_builder = StateGraph(State)

# --------- 1) Planner Agent Node ---------
def planner_node(state: State) -> dict:
    """
    Planner reads the latest user message and decides whether
    to route to the Tool Agent or the General Agent.
    """
    last_message = state["messages"][-1]
    last_text = getattr(last_message, "content", None)
    if last_text is None and isinstance(last_message, (list, tuple)) and len(last_message) > 1:
        last_text = last_message[1]
    if last_text is None:
        last_text = ""

    planner_system_prompt = """
You are the Planner Agent in a multi-agent system.

Your job:
- Decide if the user's query needs TOOLS (Wikipedia or Python REPL),
  or if a GENERAL response is enough.

Rules:
- If the user asks for factual information, current events, definitions,
  history, or "what is / who is / when is / where is", choose TOOLS.
- If the user asks for calculations, code, Python, math, or anything that
  needs executing code, choose TOOLS.
- If the user is just chatting, asking for advice, explanations, or
  non-factual creative content, choose GENERAL.

Respond with exactly ONE WORD:
- "tools"
- "general"
"""

    planner_input = [
        ("system", planner_system_prompt),
        ("user", f"User query: {last_text}")
    ]

    decision = llm.invoke(planner_input).content.strip().lower()
    if "tool" in decision:
        route = "tools"
    else:
        route = "general"

    if DEBUG_MODE:
        print(f"[Planner Agent] User query: {last_text}")
        print(f"[Planner Agent] Routing decision: {route}")

    # We update both route and agent (for debugging/trace)
    return {"route": route, "agent": "Planner"}

def route_from_planner(state: State) -> str:
    route = state.get("route") or "general"
    if route == "tools":
        return "tool_agent"
    else:
        return "general_agent"

graph_builder.add_node("planner", planner_node)

# --------- 2) Tool Agent Node ---------
def tool_agent_node(state: State) -> dict:
    """
    Tool Agent:
    - Has access to Wikipedia + Python REPL via tools.
    - Decides when to call tools using the bound-tool LLM.
    """
    tool_agent_system_prompt = """
You are the Tool Agent in a multi-agent system.

You have access to:
- Wikipedia (for factual & knowledge questions)
- Python REPL (for calculations, code, and data processing)

Instructions:
- Use tools when they genuinely help give a more accurate or useful answer.
- For factual questions: prefer Wikipedia.
- For math/code: prefer Python REPL.
- Once you have enough information, reply to the user in natural language.
"""
    messages = [("system", tool_agent_system_prompt)] + state["messages"]

    if DEBUG_MODE:
        print("[Tool Agent] Invoked with latest user/agent messages.")

    ai_msg = llm_with_tools.invoke(messages)

    if DEBUG_MODE:
        print("[Tool Agent] Produced a response (possibly after tool calls).")

    # Set agent name so the endpoint can return it
    return {"messages": [ai_msg], "agent": "Tool Agent"}

graph_builder.add_node("tool_agent", tool_agent_node)

# Shared ToolNode for Wikipedia + Python REPL
graph_builder.add_node("tools", ToolNode(tools=[wiki_tool, repl_tool]))

# --------- 3) General Agent Node ---------
def general_agent_node(state: State) -> dict:
    """
    General Agent:
    - No tools.
    - Just answers like a normal helpful assistant.
    """
    general_system_prompt = """
You are the General Agent in a multi-agent system.

You DO NOT have access to external tools.
You should:
- Chat naturally.
- Explain concepts.
- Help with reasoning, summaries, and guidance.
- If the user clearly needs live data, Wikipedia, or Python execution,
  you may say that another agent (Tool Agent) can handle that,
  but do NOT actually try to call tools yourself.
"""
    messages = [("system", general_system_prompt)] + state["messages"]

    if DEBUG_MODE:
        print("[General Agent] Invoked for normal conversational query.")

    ai_msg = llm.invoke(messages)

    if DEBUG_MODE:
        print("[General Agent] Produced a conversational response.")

    return {"messages": [ai_msg], "agent": "General Agent"}

graph_builder.add_node("general_agent", general_agent_node)

# --------- Wiring the graph ---------
graph_builder.add_edge(START, "planner")
graph_builder.add_conditional_edges("planner", route_from_planner)

# Tool Agent can optionally call tools
graph_builder.add_conditional_edges("tool_agent", tools_condition)
graph_builder.add_edge("tools", "tool_agent")

# End conditions
graph_builder.add_edge("general_agent", END)
graph_builder.add_edge("tool_agent", END)

graph = graph_builder.compile()

# ---------- LangGraph visualization ----------
try:
    image_data = graph.get_graph().draw_mermaid_png()
    image = Image.open(io.BytesIO(image_data))
    image.save("langgraph_visualization.png")
except Exception as e:
    print("Could not generate LangGraph visualization:", e)

# ---------- FastAPI endpoints ----------

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    question = request.message

    events = graph.stream(
        {"messages": [("user", question)]},
        stream_mode="values"
    )

    response = ""
    last_agent = None

    for event in events:
        # event is the current State snapshot
        response = event["messages"][-1].content
        if "agent" in event and event["agent"] is not None:
            last_agent = event["agent"]

    if DEBUG_MODE:
        print(f"[API] Final responding agent: {last_agent}")
        print(f"[API] Response: {response}")

    # Return both response text and which agent handled it
    return {"response": response, "agent": last_agent or "Unknown"}

@app.get("/")
async def serve_html():
    return FileResponse("chatbot_ui.html")
