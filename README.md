# LangChain Basics 

### 1. Build a Simple LLM Application - SimpleLLM.py

This project provides a basic introduction to building a simple LLM application using LangChain. The application translates text from English to another language through a single LLM call combined with effective prompting. While simple, this serves as an excellent starting point with LangChain, showcasing how much can be achieved with just prompting and an LLM call!

### 2. Build a Simple Chatbot with Memory - Simple_Chatbot.py

This project helps to design and implement an LLM-powered chatbot. This chatbot will be able to have a conversation and remember previous interactions with a chat model.

### 3. Build a Simple Chatbot with Prompt Templates - Simple_Chatbot_Prompt_Template.py

This project helps to design and implement an LLM-powered chatbot. We will send the messages to chatbot using Prompt Templates.

### 4. Build a Simplae Chatbot with Chat History and Stream - Simple_Chatbot_ChatHistory_Streaming.py

This project helps to design and implement an LLM-powered chatbot. We will manage conversation history and also use streaming to improve the user experience.

### 5. Build an Simple Agent - basic_agent.py

We will build an agent with the help of built-in react_agent and AgentExecutor in langchain. We will able to ask this agent a questions, it will call the datetime tool and reply with the answer.

### 6. Build an Agent - Build_Agent.py

Let's build an Agent. We will build an agent that can interact with a search engine, we will able to ask this agent questions, watch it call the search tool, and have conversations with it.

### 7. Support Chatbot using LangGraph - support_basic_chatbot_langgraph.py

Let's build our first chatbot using LangGraph. This bot can engage in basic conversation by taking user input and generating responses using an LLM.

### 8. Enhancing the Support Chatbot with Tools - support_chatbot_langgraph_with_tools.py

To handle queries our chatbot can't answer "from memory", we'll integrate a web search tool. Our bot can use this tool to find relevant information and provide better responses.

### 9. Support Chatbot with Memory - support_chatbot_langgraph_with_memory.py

Our chatbot can use tools to answer user questions, but it doesn't remember the context of previous interactions. Let's provide memory to our agent through persistent checkpointing

### 10. Support Chatbot with Human in Loop - support_chatbot_langgraph_with_human_in_loop.py

Agents can be unreliable and may need human input to successfully accomplish tasks. Similarly, for some actions, we may want to require human approval before running to ensure that everything is running as intended.
