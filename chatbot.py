import os
from api_key import api_key

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import LLMChain
os.environ["OPENAI_API_KEY"] = api_key

menu_file = "menu.md"
loader = UnstructuredMarkdownLoader(menu_file)
data = loader.load()

index = VectorstoreIndexCreator().from_loaders([loader])


st.title("Hello, I'm MenuWhiz, how may I serve you today?")
st.subheader("Type in your query below: ")
prompt_template = PromptTemplate(
    input_variables = ['topic'],
    template="Show me full details about {topic}",
)

llm = OpenAI(temperature=0)


topic = st.text_input("e.g. I want to see the menu")

if topic:
    response = index.query(topic, llm, verbose=True)
    print(response)   
    st.write(response)