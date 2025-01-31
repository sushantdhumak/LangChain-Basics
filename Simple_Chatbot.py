# ************************************************
# Build a Simplae Chatbot with message history
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
# Let's create a simple message using HumanMessage
# -----------------------------------------------

from langchain_core.messages import HumanMessage

response = model.invoke([HumanMessage(content="Hi! I am Sushant")])

print(response.content)
print(response.response_metadata)


# -----------------------------------------------
# Does model support chat history? No
# -----------------------------------------------

response = model.invoke([HumanMessage(content="What's my name?")])

print(response.content)
print(response.response_metadata)


# -----------------------------------------------
# Let's pass all message back to the model
# -----------------------------------------------

from langchain_core.messages import AIMessage, HumanMessage

response = (model.invoke(
    [
        HumanMessage(content="Hi! I'm Sushant"),
        AIMessage(content="Hello Sushant! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
))

print(response.content)
print(response.response_metadata)


# -----------------------------------------------
# Message Persistence using LangGraph
# -----------------------------------------------

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Define a graph
graph = StateGraph(state_schema=MessagesState)

# Function to call a Model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Define a node to call the model
graph.add_node("model", call_model)
graph.add_edge(START, "model")

# Add Memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# -----------------------------------------------
# Create a config 
# To support multiple conversation threads
# -----------------------------------------------

config = {"configurable": {"thread_id": "abc123"}}


# -----------------------------------------------
# Let's start with chat using above thread
# -----------------------------------------------

from langchain_core.messages import HumanMessage

query = "Hi! I'm Sushant"

input_message = [HumanMessage(content=query)]
output = app.invoke({"messages": input_message}, config=config)
output["messages"][-1].pretty_print()


query = "What's my name?"

input_message = [HumanMessage(content=query)]
output = app.invoke({"messages": input_message}, config=config)
output["messages"][-1].pretty_print()


# -----------------------------------------------
# Let's ask a question on different thread
# -----------------------------------------------

config = {"configurable": {"thread_id": "abc456"}}

input_message = [HumanMessage(content=query)]
output = app.invoke({"messages": input_message}, config=config)
output["messages"][-1].pretty_print()


# -----------------------------------------------
# Let's go back to our original thread
# -----------------------------------------------

config = {"configurable": {"thread_id": "abc123"}}

input_message = [HumanMessage(content=query)]
output = app.invoke({"messages": input_message}, config=config)
output["messages"][-1].pretty_print()


# # -----------------------------------------------