import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="RAG Question Answering System", page_icon="📄")
st.title("📄 Document Question Answering System (RAG)")

gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

if not gemini_api_key:
    st.info("Please add your Gemini API Key to proceed.", icon="🔑")
    st.stop()

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


embedding_model = load_embeddings()

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file is not None:

    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document chunks and creating vector index..."):
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        db = FAISS.from_documents(chunks, embedding_model)

    st.success("Document successfully processed and indexed!")
    os.remove(temp_file_path)  # Clean up file system

    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Generating answer..."):

            results = db.similarity_search(query, k=4)
            context = "\n".join([doc.page_content for doc in results])


            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                temperature=0,
                google_api_key=gemini_api_key
            )

            prompt = f"""
            Answer only from the context. Format the output cleanly using Markdown bullet points and bold headers. 
            Do not include raw data or metadata structures.

            Context:
            {context}

            Question: {query}
            """

            answer = llm.invoke(prompt)


            if isinstance(answer.content, list) and len(answer.content) > 0:
                text_output = answer.content[0].get('text', '')
            else:
                text_output = answer.content

            st.markdown("### 🤖 Answer:")
            st.write(text_output)