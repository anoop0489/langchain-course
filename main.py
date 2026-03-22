import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Python 101: This function looks for a file named '.env' in your folder and 
# loads the hidden variables (like your OpenAI API key) into your system's memory.
load_dotenv()

def main():
    print("Hello from langchain-course!")
    
    # Python 101: Triple quotes (""") allow you to create long, multi-line string variables.
    information = """
    Anoop is a Principal/Senior Software Engineer with 12+ years of experience designing scalable, reliable, and maintainable software systems. Passionate about leveraging agile methodologies and cloud platforms to deliver innovative solutions to complex challenges.
    🔹 Professional Expertise:
    Front-End Development: Proficient in TypeScript, React, Angular, CSS, and various CSS frameworks like Bootstrap with a strong ability to create dynamic, responsive, and visually appealing user interfaces.
    Back-End Development: Skilled in C#, Java, Golang, Python, and Node.js, with extensive experience in building robust and efficient server-side applications using technologies like ASP.NET MVC, .NET Core, Web API/REST API, and gRPC.
    Distributed Systems: Experienced in designing and implementing distributed systems using technologies like Node.js, gRPC, ASP.NET Core, Kubernetes, Docker, Apache Kafka, RabbitMQ, and Redis. Proficient in building scalable, reliable, high-performance systems that leverage microservices architecture, containerization, orchestration, message queues, and distributed databases to handle large-scale, complex applications.
    Mobile Development: Experienced in Android development, delivering high-quality mobile applications.
    Cloud Development: AWS Certified, with hands-on experience in Azure and distributed systems, leveraging cloud platforms to enhance scalability and performance.
    Event-Driven Programming: Expertise in designing and implementing event-driven architectures, utilizing technologies such as AWS SQS, AWS SNS, and AWS Lambda to build highly responsive and scalable systems.
    Databases: Proficient in SQL and NoSQL databases with expertise in ensuring efficient data management and retrieval. Skilled in using ORM frameworks such as EF & EFCore and Dapper to streamline database operations and enhance productivity
    Caching Technologies: Experienced with Distributed caching solutions such as Redis and Memcached to improve application performance and scalability.
    DevOps: Expertise in continuous integration and deployment, ensuring smooth and efficient delivery pipelines.
    🎓 Academic Background:
    Master's degree in Computer Science from the University of New Orleans.
    """

    # LangChain: This is your raw prompt string. 
    # The '{information}' acts as a placeholder where the resume data will be injected later.
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    # LangChain: We use `PromptTemplate` here because we are passing a single, simple block of text, 
    # not a structured conversation with System/Human roles.
    # Interview Pro-Tip: "I use standard PromptTemplates for simple, single-turn extraction or summarization tasks."
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # LangChain: Instantiating the AI model object. 
    # Temperature=0 makes the AI strict and factual, preventing it from making up fake skills.
    # Python 101 Note: I changed "gpt-5" to "gpt-4o" because gpt-5 is not a valid API model name yet.
    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    llm = ChatOpenAI(temperature=0, model="gpt-4o") 
    
    # LangChain: The LCEL (LangChain Expression Language) pipeline. 
    # The '|' operator connects the template directly to the AI model. 
    # Data flows from left to right.
    chain = summary_prompt_template | llm

    # LangChain: The `.invoke()` method starts the chain. 
    # It takes the 'information' variable and swaps it into the '{information}' placeholder in the template.
    response = chain.invoke(input={"information": information})
    
    # Python 101: The AI returns a complex object with lots of hidden data. 
    # Using `.content` extracts just the raw text summary we actually want to read.
    print(response.content)

if __name__ == "__main__":
    main()