import os
from api_key import api_key

import streamlit as st

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA

# Set up OpenAI
os.environ["OPENAI_API_KEY"] = api_key
llm = OpenAI(temperature=0)

# Load Data
menu_file = "menu.md"
loader = UnstructuredMarkdownLoader(menu_file)
data = loader.load()

print(f'You have {len(data)} documents in your menu.')
print(f'There are {len(data[0].page_content)} characters in the first document.')

# Chunk in smaller documents
markdown_splitter = MarkdownTextSplitter(chunk_size=450, chunk_overlap=100)
docs = markdown_splitter.create_documents([data[0].page_content])

# Create embeddings - transforming our data Documents into vectors
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(docs, embeddings)

print(f'Now you now have {len(docs)} documents in your menu.')

# Create a chain
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), verbose=True)

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables = ['topic'],
    template="Act as MenuWhiz, an AI powered Waiter and answer me this question about our burgers restaurant called 'El Gallo Gris Burgers':  {topic}. Always suggest the best option the customer, based on the query. Remember to introduce yourself at the first interaction and remain available for future questions. Always show the detailed output as it is formatted in the menu, with line breaks.",
)

# Bootstrap app with a title and a form
bg_image = "https://images.unsplash.com/photo-1572802419224-296b0aeee0d9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2015&q=80"
def add_bg_from_url():
    st.write(
         f"""
         <style>
         .stApp {{
             background-image: url({bg_image});
             background-attachment: fixed;
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
         }}
         .stApp:before {{
                content: "";
                background: rgb(14,13,13);
                background: linear-gradient(0deg, rgba(14,13,13,1) 43%, rgba(15,15,15,0.4009978991596639) 88%, rgba(25,25,25,0.7511379551820728) 100%);
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }}
           #el-galo-gris-burgers {{
            text-shadow: 2px 2px #000000;
           }} 
            
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url() 
st.title("🐓:hamburger: El Gallo Gris Burgers :hamburger:🐓")
st.subheader("Hello, I'm MenuWhiz, your AI powered waiter. How may I serve you today?")

# Create a form for the user to input a query
topic = st.text_input("Type your query. e.g. I want to see the menu")

# Run the chain and show the response
if topic:
    with st.spinner("MenuWhiz is working on your request..."):
        response = chain.run(prompt_template.format(topic=topic))
    # show loading message while waiting for response
        print(response)   
        st.write(response, unsafe_allow_html=True)