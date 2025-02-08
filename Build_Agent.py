# ************************************************
# Build an Agent
# ************************************************

# Load the environment variables

from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------
# Let's use a built-in search tool for Tavily Search Engine
# -----------------------------------------------

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
search_result = search.invoke("What is the weather in Mumbai?")
print(search_result)

# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.

tools = [search]


# -----------------------------------------------
# Let's use a language model to call tools
# -----------------------------------------------

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

model = ChatGroq(model="llama-3.3-70b-versatile")

response = model.invoke([HumanMessage(content="How are you doing?")])
print(response.content)


# -----------------------------------------------
# Let's unable model to call tools
# -----------------------------------------------

model_with_tools = model.bind_tools(tools)


# -----------------------------------------------
# Let's first call with a simple message
# -----------------------------------------------

response = model_with_tools.invoke([HumanMessage(content="Hi, my name is Sushant")])
                                   
print(f"Content String : {response.content}")
print(f"Tool Calls : {response.tool_calls}")


# -----------------------------------------------
# Let's call with a message which needs tools
# -----------------------------------------------

response = model_with_tools.invoke([HumanMessage(content="What's the weather in Pune?")])
                                   
print(f"Content String : {response.content}")
print(f"Tool Calls : {response.tool_calls}")


# -----------------------------------------------
# Let's now create an Agent using LangGraph
# -----------------------------------------------

from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)

# Run the Agent using simple message

response = agent_executor.invoke({"messages": [HumanMessage(content="Hello, how are you doing?")]})
print(response["messages"])

# Run the Agent using a message that needs tools

response = agent_executor.invoke({"messages": [HumanMessage(content="What's the weather in Pune?")]})
print(response["messages"])


# -----------------------------------------------
# Let's stream the Agent response messages
# -----------------------------------------------

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="What's the weather in Pune?")]},
):
    print(chunk)
    print("----")


# -----------------------------------------------
# Let's give memory to our Agent
# -----------------------------------------------

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

agent_executor = create_react_agent(tools=tools, model=model, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

# Let's ask a simple message

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="Hi, I am Sushant")]}, config=config
):
    print(chunk)
    print("----")

# Let's ask a my name to check the Agent memory

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="What is my name?")]}, config=config
):
    print(chunk)
    print("----")


# -----------------------------------------------
# Let's start with a new conversation with the Agent
# -----------------------------------------------

config = {"configurable": {"thread_id": "xyz123"}}

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="What is my name?")]}, config=config
):
    print(chunk)
    print("----")


# -----------------------------------------------