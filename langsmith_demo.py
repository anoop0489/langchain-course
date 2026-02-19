import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv()
# Technical Definition: A Function. It reads key-value pairs from a .env file and adds them to os.environ.
# Why: This keeps secrets (API Keys) out of your code. It also triggers LangSmith tracing
# if LANGCHAIN_TRACING_V2=true is found in your environment.


def main():

    # 2. Verify Tracing Status (Good practice for development)
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print(
            "✅ LangSmith Tracing is ENABLED. Traces will be logged to your dashboard."
        )
    else:
        print("⚠️ Tracing is DISABLED. Set LANGCHAIN_TRACING_V2=true in .env")

    # 3. DEFINE THE DATA
    information = "LangChain is a framework for developing applications powered by language models."
    # Technical Definition: A standard Python string variable representing our dynamic user data.

    # 4. DEFINE THE MESSAGE TEMPLATE (THE "BLUEPRINT")
    # CRITICAL: We use TUPLES here, not Classes/Objects.
    messages = [
        (
            "system",
            "You are a helpful AI tutor. Summarize the following concept in exactly one sentence.",
        ),
        ("human", "{information}"),
    ]
    # Technical Distinction:
    # - ("human", "{information}") is a SCHEMA (a template tuple). It is NOT a HumanMessage object yet.
    # - Using an Object like 'HumanMessage(content="{information}")' would fail here because
    #   it "locks" the text in as literal string data before the variable can be injected.

    # 5. INSTANTIATE THE CHAT PROMPT TEMPLATE
    chat_template = ChatPromptTemplate.from_messages(messages)
    # Technical Definition: 'ChatPromptTemplate' is the Class. '.from_messages()' is a Class Method.
    # 'chat_template' is the Object.
    # Why: It acts as the "Engine" that will later combine your blueprint tuples with your string data.

    # 6. INSTANTIATE THE CHAT MODEL (THE "ENGINE")
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    # Technical Definition: 'ChatOpenAI' is the Class; 'llm' is the Object (Instance).
    # - Temperature=0: Sets the model to be 'deterministic' (consistent, not creative).
    # - Model="gpt-4o": Specifies the version of the neural network to be used.

    # 7. INSTANTIATE THE OUTPUT PARSER (THE "CLEANER")
    parser = StrOutputParser()
    # Technical Definition: 'StrOutputParser' is the Class. 'parser' is the Object.
    # Why: It intercepts the AI's complex 'AIMessage' object and extracts only the 'content' string.
    # Without this, you get a JSON-like object back.

    # 8. BUILD THE LCEL CHAIN (THE "PIPELINE")
    chain = chat_template | llm | parser
    # Technical Definition: LCEL (LangChain Expression Language) uses the pipe operator '|'.
    # Python Secret: The "|" symbol is actually a Python feature called "Operator Overloading".
    # It triggers a hidden dunder (double underscore) method called `__or__`.
    # It tells Python to pass the output of the left object into the input of the right object.

    # 9. EXECUTION (THE "START" BUTTON)
    response = chain.invoke(input={"information": information})
    # Technical Definition: .invoke() is the Method that triggers the chain.
    # Behind the scenes:
    #   1. LangChain finds the "{information}" placeholder in your tuple.
    #   2. It injects the 'information' string.
    #   3. It finally creates the 'HumanMessage' OBJECT and sends it to the AI.
    #   4. It sends a 'trace' to LangSmith asynchronously for debugging.

    # 10. OUTPUT
    print("OUTPUT:*****************************************************")
    print(response)


if __name__ == "__main__":
    main()
