from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama ## all third-party integrations are available in langchain_community

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# get environment variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "True"

# prompt template
prompts = ChatPromptTemplate(
    [
        ("system", "You are a chatbot assistant. Kindly answer the user queries."),
        ("user", "Question: {question}")
    ]
)

# call Ollama LLM model
llm_model = Ollama(model="LLAMA2")

# define output parser
output_parser = StrOutputParser()

# define the chain
chain = prompts|llm_model|output_parser

# streamlit framework
st.title("LangChain based Chatbot using Ollama (LLAMA2)")
input_text = st.input_text("Enter your query here")

if input_text:
    st.write(chain.invoke({"question": input_text}))





