# ************************************************
# Support ChatBot with Human in Loop using LangGraph
# ************************************************

# Load the environment variables

from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------
# Build a Basic Chatbot
# -----------------------------------------------

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(state_schema=State)


# -----------------------------------------------
# Let's add a 'Chatbot' node
# -----------------------------------------------

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")


# -----------------------------------------------
# Enhancing the ChatBot with Tools
# -----------------------------------------------

from langchain_core.tools import tool
from langgraph.types import Command, interrupt

@tool
def human_assistance(query:str) -> str:
    """
    Request assistance from a Human.
    """
    human_response = interrupt({"query" : query})
    return human_response["data"]


from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
tools = [search, human_assistance]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state:State):
    message =  llm_with_tools.invoke(state["messages"])

    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages" : [message]}

graph_builder.add_node("chatbot", chatbot)


# -----------------------------------------------
# Let's use LangGraph's prebuilt ToolNode and tools_condition
# -----------------------------------------------

from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools=tools)


# -----------------------------------------------
# Let's add graph nodes and edges
# -----------------------------------------------

graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")


# -----------------------------------------------
# Let's compile graph using checkpointer
# -----------------------------------------------

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# -----------------------------------------------
# Let's Visualize the graph
# -----------------------------------------------

from IPython.display import Image, display

try:
    display(Image (graph.get_graph().draw_mermaid_png()))
except Exception:
    pass


# -----------------------------------------------
# Let's Run the ChatBot
# -----------------------------------------------

# create a thread id

config = {"configurable" : {"thread_id" : "123"}}

# Let's give an input to the graph

input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"

events = graph.stream(
    {"messages" : [{"role" : "user", "content" : input}]},
    config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


# -----------------------------------------------
# Human response using Command
# -----------------------------------------------

human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    "It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data" : human_response})

events = graph.stream(human_command, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


# -----------------------------------------------