# Loads environment variables from .env file
# Use case: Keeps API keys secret (best practice)
from dotenv import load_dotenv
import os
import streamlit as st

# Load .env variables into the environment
load_dotenv()


# ChatGroq = LangChain wrapper for Groq-hosted LLMs
# Use case: Connects app to Groq models like LLaMA 3
from langchain_groq import ChatGroq


# ChatPromptTemplate = structured prompt builder
# Use case: Helps to define system + user messages cleanly
from langchain_core.prompts import ChatPromptTemplate


# StrOutputParser = converts model output into plain text
# Use case: Makes sure output is clean string (not JSON/objects)
from langchain_core.output_parsers import StrOutputParser

# environment variables call
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
##langsmith tracking for debugging monitoring and evaluation and tracking
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = (
    "true"  # It enables tracing of LangChain runs to visualize execution flow, debug issues, and monitor performance.
)


##creating chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please provide response to the user queries accurately.",
        ),
        ("user", "Question: {question}"),
    ]
)

##streamlit framework for web app
st.title("📚 Groq + LangChain Chatbot")
input_question = st.text_input("Search for your topic:")

##open AI llm model from groq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
output_parser = StrOutputParser()

##chain creation
chain = prompt | llm | output_parser
##usecase of chain is to process user input, generate response using llm and parse the output to string
if input_question:
    st.write(chain.invoke({"question": input_question}))
