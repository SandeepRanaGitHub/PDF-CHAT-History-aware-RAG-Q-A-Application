import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# environment setup
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_Token"] = os.getenv("HF_Token")
# embedding 
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

st.title("PDF-Chat History aware RAG Q&A Application")
st.write("Upload PDF filesand chat with their content")

# Api Key
api_key = st.text_input("Enter your GROQ API KEY", type="password")
## Check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="openai/gpt-oss-120b")

    # Chat Interface
    session_id = st.text_input("Session ID",value="default_session")
    # Statefully manage  chat history
    if 'store' not in st.session_state:
        st.session_state.store={}
    #upload files
    uploaded_files = st.file_uploader("Upload PDF files",accept_multiple_files=True, type='pdf')

    # prosess uploaded file
    if uploaded_files:
        document = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            document.extend(docs)
        # Split and creat embedding for document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        split = text_splitter.split_documents(document)
        vectorstore = Chroma.from_documents(documents=split, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which mmight reference context in the chat history"
            "formulate a standalone question which can be understood without the chat history"
            "Do nOt answer it just reformulate it if needed and otherwise return it as it."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human","{input}")
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        # answer question
        system_prompt = (
            "your are a helpful assistant for question answering task "
            "use the following piece of reterived context to answer"
            "the question.If you dont know just sap i dont know"
            "{context}"
        )
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human","{input}")
            
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm,answer_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            output_messages_key="answer",
            history_messages_key="chat_history"
        )

        user_input = st.text_input("Please Ask any Question")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config ={"configurable":{"session_id":session_id}}
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.write("Please Enter Your Api Key")