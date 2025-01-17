# Imports
import os 
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message
import json
import openai

# Set API keys and the models to use
# Open the JSON file
with open('/Users/edwin/Repositories/api_keys.json') as file:
    data = json.load(file)

# Access the API keys
openai.api_key= data['apiKeys']['OpenAI']

os.environ["OPENAI_API_KEY"] = data['apiKeys']['OpenAI']

model_id = "gpt-3.5-turbo"

# Load PDF document
# Langchain has many document loaders 
# We will use their PDF load to load the PDF document below
loaders = PyPDFLoader('/Users/edwin/Repositories/Chatgpt/sample_documents/ug_cr_rptstd.pdf')

# Create a vector representation of this document loaded
index = VectorstoreIndexCreator().from_loaders([loaders])


st.title('🦜 Query your PDF document ')
prompt = st.text_input("Enter your question to query your PDF documents")


#------------------------------------------------------------------
# Save history

# This is used to save chat history and display on the screen
if 'answer' not in st.session_state:
    st.session_state['answer'] = []

if 'question' not in st.session_state:
    st.session_state['question'] = []    



# Display the current response. No chat history is maintained

if prompt:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM
    response = index.query(llm=OpenAI(model_name = model_id, temperature=0.2), question = prompt, chain_type = 'stuff')

    # Add the question and the answer to display chat history in a list
    # Latest answer appears at the top
    st.session_state.question.insert(0,prompt  )
    st.session_state.answer.insert(0,response  )            
    
    # Display the chat history
    for i in range(len( st.session_state.question)) :
        message(st.session_state['question'][i], is_user=True)
        message(st.session_state['answer'][i], is_user=False)