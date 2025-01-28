# ************************************************
# Build a Simple LLM Application with LCEL
# ************************************************

from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------
# LangSmith to set your environment variables 
# to start logging traces using Langsmith API key
# ------------------------------------------------

import os

os.getenv("OPENAI_API_KEY")

# os.environ["LANGCHAIN_TRACING_V2"] = ""
# os.environ["LANGCHAIN_API_KEY"] = "" 


# -----------------------------------------------
# Using Language Models
# -----------------------------------------------

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")


from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="How are you?"),
]

model.invoke(messages)
print(model.invoke(messages))


# -----------------------------------------------
# OutputParsers
# -----------------------------------------------

from langchain_core.output_parsers import StrOutputParser  

parser = StrOutputParser()
result = model.invoke(messages)
parser.invoke(result)
print(parser.invoke(result))


# -----------------------------------------------
# Chain
# -----------------------------------------------

chain = model | parser
print(chain.invoke(messages))


# -----------------------------------------------
# Prompt Templates
# -----------------------------------------------

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

result = prompt_template.invoke({"language" : "italian", "text" : "hello"})
result.to_messages()
print(result.to_messages())

response = model.invoke(result.to_messages())
print(response)


# -----------------------------------------------
# Chaining together components with LCEL
# -----------------------------------------------

chain = prompt_template | model | parser

# chain.invoke({"language" : "italian", "text" : "hi" })
print(chain.invoke({"language" : "italian", "text" : "hi" }))

