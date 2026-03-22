import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES & TRACING
# ==========================================

# Python 101: Loads hidden environment variables from a local .env file into os.environ.
# LangChain: This automatically triggers LangSmith tracing in the background if LANGCHAIN_TRACING_V2=true is set in your .env.
load_dotenv()

def main():
    # Python 101: os.getenv() retrieves the value of a specific environment variable safely.
    # Pro-Tip: "I always add a tracing status check at the start of my scripts so I can verify my observability pipeline is active before running expensive LLM calls."
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("✅ LangSmith Tracing is ENABLED. Traces will be logged to your dashboard.")
    else:
        print("⚠️ Tracing is DISABLED. Set LANGCHAIN_TRACING_V2=true in .env")

    # ==========================================
    # 2. DEFINE THE DATA & TEMPLATE ("BLUEPRINT")
    # ==========================================
    
    # Python 101: A standard string variable representing the dynamic data the user will provide.
    information = "LangChain is a framework for developing applications powered by language models."

    # LangChain: We use TUPLES for dynamic templates, not HumanMessage/SystemMessage objects.
    # Pro-Tip: "Using tuples creates a 'schema' that waits for data. If you use a Message object here, it locks the text as literal string data and your variable injection will fail."
    messages = [
        ("system", "You are a helpful AI tutor. Summarize the following concept in exactly one sentence."),
        ("human", "{information}"),
    ]

    # ==========================================
    # 3. INSTANTIATE THE COMPONENTS
    # ==========================================
    
    # Python 101: '.from_messages()' is a Class Method acting as an alternative constructor.
    # LangChain: This builds the "Engine" that will eventually combine your blueprint tuples with your string data.
    chat_template = ChatPromptTemplate.from_messages(messages)

    # LangChain: Instantiates the AI engine. 
    # Pro-Tip: "Setting the temperature to 0 makes the model deterministic, which is the industry standard for strict summarization and data extraction tasks."
    llm = ChatOpenAI(temperature=0, model="gpt-4o")

    # LangChain: Intercepts the AI's complex 'AIMessage' object and extracts only the readable 'content' string so you don't have to deal with the metadata payload.
    parser = StrOutputParser()

    # ==========================================
    # 4. BUILD THE LCEL CHAIN ("PIPELINE")
    # ==========================================
    
    # LangChain: LCEL (LangChain Expression Language) uses the pipe operator '|' to flow data left to right.
    # Python 101: The "|" symbol triggers "Operator Overloading" via the hidden `__or__` dunder method, telling Python to pass output from the left component into the right component.
    chain = chat_template | llm | parser

    # ==========================================
    # 5. EXECUTION ("START" BUTTON)
    # ==========================================
    
    # LangChain: .invoke() injects the dictionary data into the "{information}" placeholder, creates the final Message objects, and sends them through the LLM and Parser.
    response = chain.invoke(input={"information": information})

    # Python 101: Prints the final string that was cleanly extracted by the StrOutputParser.
    print("OUTPUT:*****************************************************")
    print(response)

if __name__ == "__main__":
    main()