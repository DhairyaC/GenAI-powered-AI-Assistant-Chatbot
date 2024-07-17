# import all the necessary modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# get environment variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") ## tells us where the monitoring results need to be stored, i.e., the LangChain dashboard
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "True" ## to capture all the monitoring results - LangSmith (part of LangChain ecosystem)

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a chatbot assistant. Kindly answer the user queries."),
        ("user", "Question: {question}")
    ]
)

# call OpenAI LLM model
llm_model = ChatOpenAI(model="gpt-3.5-turbo")

# define output parser
output_parser = StrOutputParser()

# define the chain
chain = prompt|llm_model|output_parser

# streamlit framework
st.title("LangChain based Chatbot using OpenAI API")
input_text = st.text_input("Enter your query here")

if input_text:
    st.write(chain.invoke({"question": input_text}))

