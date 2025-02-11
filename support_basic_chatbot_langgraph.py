# ************************************************
# Support ChatBot using LangGraph
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

def chatbot(state:State):
    return {"messages":  [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

# Add and Entry point

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


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

def stream_graph_updates(user_input : str):
    for event in graph.stream({"messages": [{"role" : "user", "content" : user_input}]}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        stream_graph_updates(user_input)

    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
    

# -----------------------------------------------