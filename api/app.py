from fastapi import FastAPI
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langserve import add_routes

# add_routes belongs to LangServe

# LangServe is a LangChain ecosystem library

# It integrates with FastAPI, but FastAPI does not provide add_routes
import uvicorn
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)
add_routes(
    app,
    ChatGroq(model="llama-3.3-70b-versatile", temperature=0),
    path="/groq-chat",  # The endpoint path for the Groq Chat model
)
# groq llm model instance
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
output_parser = StrOutputParser()
prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Write me a poem about {topic}. with rhymes and 100 words"),
    ]
)
prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Write a detailed explanation about {concept} in simple terms."),
    ]
)


add_routes(
    app,
    prompt1 | model | output_parser,
    path="/poem",  # The endpoint path for the poem generator
)

add_routes(
    app,
    prompt2 | model | output_parser,
    path="/concept",  # The endpoint path for the concept explainer
)
# output_parser1 = StrOutputParser()
# chain1 = prompt1 | model | output_parser1
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
