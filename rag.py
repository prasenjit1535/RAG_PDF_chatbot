from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")  # e.g., "sentence-transformers/all-MiniLM-L6-v2"

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

from htmlTemplates import bot_template, user_template, css

# -------------------- PDF & Text Chunking --------------------

def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# -------------------- Embedding + Vector Store --------------------

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# -------------------- LLM Chain Setup --------------------

def get_conversation_chain(vector_store):
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL_NAME, temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    system_template = """
    Use the following pieces of context and chat history to answer the question at the end.
    If you don't know the answer, just say you don't know, don't make anything up.

    Context: {context}
    Chat history: {chat_history}
    Question: {question}
    Helpful Answer:
    """

    prompt = PromptTemplate(
        template=system_template,
        input_variables=["context", "question", "chat_history"],
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True
    )
    return conversation_chain

# -------------------- Chat Handler --------------------

def handle_user_input(question):
    try:
        response = st.session_state.conversation.invoke({'question': question})
        st.session_state.chat_history = response['chat_history']
        
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", response['answer']), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# -------------------- UI --------------------

def main():
    st.set_page_config(page_title='Chat with PDFs', page_icon='📄')
    st.write(css, unsafe_allow_html=True)
    st.header('📄 Chat with PDFs')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input box for question
    question = st.text_input("Ask anything to your PDF:")

    if question:
        if st.session_state.conversation:
            handle_user_input(question)
        else:
            st.warning("⚠️ Please upload and process a PDF first.")

    # Sidebar - PDF Upload
    with st.sidebar:
        st.subheader("Upload your Documents:")
        pdf_files = st.file_uploader("Choose PDF(s) & press Process", type=['pdf'], accept_multiple_files=True)

        if pdf_files and st.button("Process"):
            with st.spinner("🔄 Processing PDFs..."):
                try:
                    raw_text = get_pdf_text(pdf_files)
                    if not raw_text.strip():
                        st.error("❌ Could not extract text from PDF. Please try another file.")
                        return
                    
                    chunks = get_chunk_text(raw_text)
                    vector_store = get_vector_store(chunks)

                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("✅ PDFs processed! You can now ask questions.")
                except Exception as e:
                    st.error(f"❌ Error while processing PDFs: {e}")

if __name__ == '__main__':
    main()
