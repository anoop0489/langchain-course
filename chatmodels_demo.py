import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Python 101: We import specific message classes to strictly define "who" is speaking in the code.
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. Load Config
# Python 101: Loads hidden environment variables (like OPENAI_API_KEY) from a local .env file.
load_dotenv()

# 2. Initialize the Chat Model
# LangChain: We use a 'Chat' model instead of a standard base 'LLM'. 
# Python 101 Note: Changed "gpt-5" to "gpt-4o" to use a valid, active OpenAI model.
# Pro-Tip: "A temperature of 0.7 allows the model to be slightly creative, which is perfect for maintaining a specific, colorful persona."
llm = ChatOpenAI(temperature=0.7, model="gpt-4o")

# 3. Define a "Conversation History" (The Context)
# LangChain: Chat models don't just take a single string of text. They require a LIST 
# of specialized message objects. This list represents the entire state of the conversation.
messages = [
    # LangChain: SystemMessage acts as the "God Mode" instructions. 
    # It defines the AI's core behavior, rules, and persona before the chat even begins.
    SystemMessage(content="You are a sarcastic senior engineer who loves Python."),
    
    # LangChain: HumanMessage represents the actual input from the end-user.
    HumanMessage(content="I am writing a script to parse CSVs."),
    
    # LangChain: AIMessage represents the AI's past responses. 
    # Pro-Tip: "By manually injecting an AIMessage into the list, we 'fake' a conversation history to show the model exactly how we want it to respond to future prompts."
    AIMessage(content="Oh, thrilling. Another CSV parser. Groundbreaking work."),
    
    # LangChain: The current/latest question from the user.
    HumanMessage(content="Hey, be nice! How do I handle missing values with pandas?"),
]

# 4. Execution
# LangChain: We pass the entire LIST of messages into the `.invoke()` method. 
# The AI reads the system rules, reviews the history, and generates the next logical AIMessage.
response = llm.invoke(messages)

# 5. Output Analysis
# Python 101: type(response).__name__ dynamically prints the name of the class (which will output 'AIMessage').
print(f"Role: {type(response).__name__}") 

# LangChain: Just like in our previous chains, we must use `.content` to extract the readable string from the complex AIMessage object.
print(f"Content: {response.content}")