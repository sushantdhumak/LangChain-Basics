from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, tool
from datetime import datetime

# -----------------------------------------------
# Loading the Environment Variables

from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------
# Defining a tool to get the date and time

@tool
def get_system_datetime(format: str = "%d-%m-%Y %H:%M:%S"):
    """
    Returns the current date and time in specified format
    """
    current_time = datetime.now()
    formatted_time = current_time.strftime(format)

    return formatted_time

# -----------------------------------------------
# LLM Model

llm = ChatOpenAI(model="gpt-4o-mini")

# -----------------------------------------------
# User query and prompt_template

# query = "What is the current date and time?"
query = "What is the current time in Auckland considering you are in India? Just show the time not date."

prompt_template = hub.pull("hwchase17/react")

# -----------------------------------------------
# Tools and Agent Execution

tools = [get_system_datetime]

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input" : query})

# -----------------------------------------------