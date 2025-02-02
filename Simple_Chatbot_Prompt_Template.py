# ************************************************
# Build a Simplae Chatbot with Prompts
# ************************************************

# Loading environment variables

from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------
# Let's use a LLM Model for Chatbot
# -----------------------------------------------

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")


# -----------------------------------------------
# Use of Prompt Template
# MessagesPlaceholder to pass all the messages in a single prompt
# -----------------------------------------------

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages(
    [
        ( 
          "system",
          "You are a helpful assistant who sounds like a pirate. Answer all questions to the best of your ability."
        ),

        MessagesPlaceholder(variable_name="messages"),
    ]
)

# -----------------------------------------------
# Message Persistence using LangGraph
# -----------------------------------------------

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Define a graph
graph = StateGraph(state_schema=MessagesState)

# Function to call a Model
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

# Define a node to call the model
graph.add_edge(START, "model")
graph.add_node("model", call_model)

# Add Memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# -----------------------------------------------
# Create a config 
# To support multiple conversation threads
# -----------------------------------------------

from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Sushant"

input_message = [HumanMessage(content=query)]
output = app.invoke({"messages": input_message}, config=config)
output["messages"][-1].pretty_print()


query = "What's my name?"

input_message = [HumanMessage(content=query)]
output = app.invoke({"messages": input_message}, config=config)
output["messages"][-1].pretty_print()


# -----------------------------------------------
# Let's try with a different prompt
# -----------------------------------------------

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
          "system",
          "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),

        MessagesPlaceholder(variable_name="messages"),
    ]
)


# -----------------------------------------------
# Message Persistence using LangGraph
# -----------------------------------------------

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

class state(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# Define a graph
graph = StateGraph(state_schema=state)

# Function to call a Model
def call_model(state: state):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

# Define a node to call the model
graph.add_edge(START, "model")
graph.add_node("model", call_model)

# Add Memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# -----------------------------------------------

config = {"configurable": {"thread_id": "abc456"}}

query = "Hi! I'm Sushant"
language = "Spanish"

input_message = [HumanMessage(content=query)]
output = app.invoke({"messages": input_message, "language": language}, config=config)
output["messages"][-1].pretty_print()


# -----------------------------------------------
# Let's go back to our original thread
# -----------------------------------------------

query = "What's my name?"

input_message = [HumanMessage(content=query)]
output = app.invoke({"messages": input_message}, config=config)
output["messages"][-1].pretty_print()


# -----------------------------------------------