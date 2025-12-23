import streamlit as st
import os
from dotenv import load_dotenv
import requests

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


def get_response(input_question):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json={"input": {"topic": input_question}},
    )
    return response.json()["output"]


def get_response1(input_question1):
    response = requests.post(
        "http://localhost:8000/concept/invoke",
        json={"input": {"concept": input_question1}},
    )
    return response.json()["output"]


st.title("Langchain demo with groq llm model")
input_question = st.text_input("Write a poem on: ")
input_question1 = st.text_input("Search for your topic:")
if input_question:
    st.write(get_response(input_question))
if input_question1:
    st.write(get_response1(input_question1))
