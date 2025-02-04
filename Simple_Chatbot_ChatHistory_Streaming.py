# ************************************************
# Build a Simplae Chatbot with Chat History and Stream
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
          "You are a helpful assistant."
        ),

        MessagesPlaceholder(variable_name="messages"),
    ]
)


# -----------------------------------------------
# Let's use trim_messsage to remove the content
# of the message from end
# -----------------------------------------------

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages 

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm Sushant"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like to code"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 3 * 4"),
    AIMessage(content="12"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

print(trimmer.invoke(messages))


# -----------------------------------------------
# Define a graph state schema
# -----------------------------------------------

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# -----------------------------------------------
# Message Persistence using LangGraph
# -----------------------------------------------

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

# Define a graph
graph = StateGraph(state_schema=State)

# Function to call a Model
def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
        )
    response = model.invoke(prompt)
    return {"messages": [response]}

# Define a node to call the model
graph.add_edge(START, "model")
graph.add_node("model", call_model)

# Add Memory
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# -----------------------------------------------
# Let's try asking model my name
# -----------------------------------------------

config = {"configurable": {"thread_id": "abc456"}}
query = "What is my name?"
language = "English"

input_message = messages + [HumanMessage(content=query)]
output = app.invoke({"messages": input_message, "language": language}, config=config)
output["messages"][-1].pretty_print()


# -----------------------------------------------
# Let's try asking model the math problem
# -----------------------------------------------

config = {"configurable": {"thread_id": "abc567"}}
query = "What was the math problem I asked?"
language = "English"

input_message = messages + [HumanMessage(content=query)]
output = app.invoke({"messages": input_message, "language": language}, config=config)
output["messages"][-1].pretty_print()


# -----------------------------------------------
# Let's create a Streaming response
# -----------------------------------------------

config = {"configurable": {"thread_id": "abc678"}}
query = "Hi, I am Sushant. PLease tell me a funny joke."
language = "English"

input_messages = [HumanMessage(content=query)]

for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language}, 
    config=config,
    stream_mode="messages",
    ):
    if isinstance(chunk, AIMessage):
        print(chunk.content, end="|")

# -----------------------------------------------