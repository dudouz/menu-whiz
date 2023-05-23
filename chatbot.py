import os
from api_key import api_key

import streamlit as st
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = api_key

menu_file = "menu.md"
loader = UnstructuredMarkdownLoader(menu_file, mode="elements")
data = loader.load()

index = VectorstoreIndexCreator().from_loaders([loader])


st.title("MenuWhiz - Your Personal Restaurant Assistant")
st.subheader("Powered by OpenAI's GPT-3, Langchain's LLMS, and Streamlit")

llms = OpenAI(temperature=0.9)

prompt = st.text_input("Hello sir, how may I serve you: ")

if prompt:
    response = index.query(prompt, llms)
    st.write(response)