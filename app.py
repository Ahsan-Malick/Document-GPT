import os
import io
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_community.llms import openai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import conversational_retrieval
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain_community.callbacks.manager import get_openai_callback
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv





def main():
    load_dotenv()
    open_ai_key  = os.environ.get("OPENAI_API_KEY")
    st.set_page_config(page_title="Talk with your own document")
    st.header("DocumentGPT")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=[
                                          "pdf", "docx"], accept_multiple_files=False)
        
        open_ai_key = open_ai_key
        process = st.button("process")

    if process:
        if not open_ai_key:
            st.info("Enter openai key")
            st.stop()

        files_text = get_files_text(uploaded_files)
        st.write("File loaded...")

        text_chunks = get_text_chunks(files_text)
        st.write("File chunks created...")

        vector_store = get_vectorstore(text_chunks)
        st.write("Vector store created...")

        st.session_state.conversation = get_conversation_chain(
            vector_store, open_ai_key)
        # st.session_state.conversation = get_conversation_chain(vector_store) for huggingface

        st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask question about your file.")
        if user_question:
            handle_userinput(user_question)

def get_files_text(uploaded_files):
    text = ""
    split = os.path.splitext(uploaded_files.name)
    file_extension = split[1]
    if file_extension == ".pdf":
        text += get_pdf_text(uploaded_files)
    else:
           print("file not supported")
    return text

    # for uploaded_file in uploaded_files:
    #     split_tup = os.path.splitext(uploaded_file.name)
    #     file_extension = split_tup[1]
    #     if file_extension == ".pdf":
    #         text += get_pdf_text(uploaded_file)
    #     else:
    #         print("file not supported")
    # return text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

# using langchain function
def get_conversation_chain(vetorestore, open_ai_key):
    llm = ChatOpenAI(openai_api_key=open_ai_key, model_name= 'gpt-3.5-turbo', temperature=0)
    # chat memory storing in chat_history key    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
             # using langchain function                       
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
)
    return conversation_chain

def handle_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
        print(response)
    st.session_state.chat_history = response['chat_history']

    response_container = st.container()

    with response_container:
        for i,messages in enumerate(st.session_state.chat_history):
            if i%2==0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

if __name__=="__main__":
    main()
