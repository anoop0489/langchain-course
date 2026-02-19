from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. Load Config
load_dotenv()

# 2. Initialize the Chat Model
llm = ChatOpenAI(temperature=0.7, model="gpt-5")

# 3. Define a "Conversation History" (The Context)
# Unlike completion models, we don't just send a query.
# We send a structured list representing the state of the conversation.
messages = [
    # SYSTEM: The "God Mode" instruction. Sets behavior/persona.
    SystemMessage(content="You are a sarcastic senior engineer who loves Python."),
    # HUMAN: The user's first input.
    HumanMessage(content="I am writing a script to parse CSVs."),
    # AI: A 'fake' history. We inject this to give the model memory of what it 'said'.
    AIMessage(content="Oh, thrilling. Another CSV parser. Groundbreaking work."),
    # HUMAN: The user's follow-up question.
    HumanMessage(content="Hey, be nice! How do I handle missing values with pandas?"),
]

# 4. Execution
# We pass the LIST of messages, not a single string.
response = llm.invoke(messages)

# 5. Output Analysis
print(f"Role: {type(response).__name__}")  # Expect: AIMessage
print(f"Content: {response.content}")
