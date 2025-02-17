# ************************************************
# Support ChatBot with Memory using LangGraph
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

# TAVILY_API_KEY:  ········

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
tools = [search]

# search.invoke("What's a 'node' in LangGraph?")

llm_with_tools = llm.bind_tools(tools)

def chatbot(state:State):
    return {"messages":  [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)


# -----------------------------------------------
# Let's use LangGraph's prebuilt ToolNode and tools_condition
# -----------------------------------------------

from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools=[search])


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

input = "Hi! My name is Sushant."

events = graph.stream(
    {"messages" : [{"role" : "user", "content" : input}]},
    config,
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()


# -----------------------------------------------

# Let's check whether bot remember my name
# Keeping the thread_id same as previous  

input = "Remember my name?"

events = graph.stream(
    {"messages" : [{"role" : "user", "content" : input}]},
    config,
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()


# -----------------------------------------------

# What happens when we change the thread_id

input = "Remember my name?"

events = graph.stream(
    {"messages" : [{"role" : "user", "content" : input}]},
    {"configurable" : {"thread_id" : "234"}},
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()


# -----------------------------------------------