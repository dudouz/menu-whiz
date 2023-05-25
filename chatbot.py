import os
from api_key import api_key

import streamlit as st

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = api_key

llm = OpenAI(temperature=0, max_tokens=5000)

menu_file = "menu.md"
loader = UnstructuredMarkdownLoader(menu_file)
data = loader.load()
markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)

docs = markdown_splitter.create_documents([data[0].page_content])

print(docs, 'docs')

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(docs, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())


st.title("El Galo Gris Burgers")
st.subheader("Hello, I'm MenuWhiz, your AI powered waiter. How may I serve you today?")
# st.subheader("Type in your query below: ")
prompt_template = PromptTemplate(
    input_variables = ['topic'],
    template="Act as MenuWhiz, an AI powered Waiter and answer me this question about our burgers restaurant called 'El Galo Gris Burgers':  {topic}. Always suggest the best option the customer, based on the query. Remember to introduce yourself at the first interaction and remain available for future questions. Always show the menu output as it is formatted, with line breaks.",
)



topic = st.text_input("Type your query. e.g. I want to see the menu")

if topic:
    with st.spinner("MenuWhiz is working on your request..."):
        response = qa.run(prompt_template.format(topic=topic))
    # show loading message while waiting for response
        print(response)   
        st.write(response)