"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) 
using HuggingFace + Gemini + LangChain.

✅ FAISS-based vector DB
✅ Multiple PDFs (cumulative)
✅ PDF preview with zoom
✅ Sidebar + 2-column UI
"""

import os
import streamlit as st
import logging
import tempfile
import shutil
import pdfplumber
import warnings
from typing import List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# ----------------- Suppress Warnings -----------------
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------- Config -----------------
MAX_PDFS = 5

# 1️⃣ Streamlit page config
st.set_page_config(
    page_title="📄 AI PDF Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2️⃣ Load .env
load_dotenv()
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found! Set it in .env or Streamlit Secrets.")
else:
    st.sidebar.success("✅ GOOGLE_API_KEY loaded.")

# ----------------- Imports for LangChain & HuggingFace -----------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import pipeline, BitsAndBytesConfig
import torch

# ----------------- Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ----------------- Utility Functions -----------------
def create_vector_db(file_uploads: List[Any]) -> FAISS:
    """Create FAISS vector store from uploaded PDFs."""
    all_chunks = []

    for file_upload in file_uploads:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                doc = Document(
                    page_content=text,
                    metadata={"source": file_upload.name, "page": i + 1}
                )
                splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
                chunks = splitter.split_documents([doc])
                all_chunks.extend(chunks)

        shutil.rmtree(temp_dir)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_db = FAISS.from_documents(all_chunks, embeddings)
    return vector_db


@st.cache_resource
def get_hf_pipeline(model_name: str):
    """Return HuggingFace pipeline LLM."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    device = 0 if torch.cuda.is_available() else -1
    task = "text2text-generation" if "t5" in model_name.lower() or "bart" in model_name.lower() else "text-generation"

    pipe = pipeline(
        task,
        model=model_name,
        tokenizer=model_name,
        max_new_tokens=512,
        device=device,
        trust_remote_code=True,
        quantization_config=quantization_config if device != -1 else None,
    )
    return HuggingFacePipeline(pipeline=pipe)


@st.cache_resource
def get_gemini_llm(model_name: str = "gemini-1.5-pro"):
    """Return Gemini LLM."""
    if not GOOGLE_API_KEY:
        raise ValueError("❌ GOOGLE_API_KEY not found.")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        max_output_tokens=512,
        google_api_key=GOOGLE_API_KEY,
        credentials=None
    )


def process_question(question: str, vector_db: Optional[FAISS], llm) -> str:
    """Process user question with RAG."""
    if vector_db is None:
        return "🤔 No documents uploaded for this chat."

    results = vector_db.similarity_search_with_score(question, k=3)
    if not results or all(not doc.page_content.strip() for doc, _ in results):
        return "I don’t know, it is not mentioned in the uploaded PDFs."

    context_texts = "\n\n".join([doc.page_content for doc, _ in results])

    template = """
Answer the question based ONLY on the following context.
If the answer is not in the context, say clearly: 
"I don’t know, it is not mentioned in the uploaded PDFs."
----------------
{context}
----------------
Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": lambda _: context_texts, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        return chain.invoke(question).strip()
    except Exception as e:
        return f"⚠️ Error: {e}"


@st.cache_data
def extract_all_pages_as_images(file_uploads: List[Any]) -> List[Any]:
    """Return list of dicts {image, source, page} for preview."""
    pdf_pages = []
    for file_upload in file_uploads:
        with pdfplumber.open(file_upload) as pdf:
            for i, page in enumerate(pdf.pages):
                pdf_pages.append({
                    "image": page.to_image().original,
                    "source": file_upload.name,
                    "page": i + 1
                })
    return pdf_pages


# ----------------- Main App -----------------
def main() -> None:
    st.markdown("# 📄 AI PDF Chat Assistant")
    st.markdown("#### Your intelligent assistant for document understanding.")

    col1, col2 = st.columns([1.5, 2])

    if "chats" not in st.session_state:
        st.session_state["chats"] = {}
    if "active_chat" not in st.session_state:
        st.session_state["active_chat"] = None
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = None
    if "llm" not in st.session_state:
        st.session_state["llm"] = None

    # ---------------- Sidebar ----------------
    st.sidebar.subheader("💬 Chats")
    if st.sidebar.button("➕ New Chat"):
        chat_id = "Chat - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["chats"][chat_id] = {
            "messages": [], "vector_db": None, "file_uploads": [], "pdf_pages": []
        }
        st.session_state["active_chat"] = chat_id

    for chat_id in list(st.session_state["chats"].keys())[::-1]:
        col_a, col_b = st.sidebar.columns([8, 1])
        if col_a.button(chat_id, key=f"open_{chat_id}"):
            st.session_state["active_chat"] = chat_id
        if col_b.button("🗑️", key=f"del_{chat_id}"):
            st.session_state["chats"].pop(chat_id, None)
            if st.session_state.get("active_chat") == chat_id:
                st.session_state["active_chat"] = None
            break

    # ---------------- Left Column ----------------
    with col1:
        st.subheader("📂 Documents & Collections")
        active = st.session_state.get("active_chat")
        if active is None:
            st.info("Start a new chat (sidebar) or select an existing chat.")
        else:
            st.markdown(f"**Active chat:** {active}")

        file_uploads = st.file_uploader(
            f"Upload PDF files (max {MAX_PDFS})",
            type="pdf",
            accept_multiple_files=True,
            key=f"uploader_{active}"
        )

        if file_uploads and active:
            existing_uploads = st.session_state["chats"][active].get("file_uploads", [])
            all_uploads = existing_uploads + file_uploads
            if len(all_uploads) > MAX_PDFS:
                st.warning(f"Max {MAX_PDFS} PDFs allowed. Extras ignored.")
                all_uploads = all_uploads[:MAX_PDFS]

            st.session_state["chats"][active]["file_uploads"] = all_uploads
            with st.spinner("Processing PDFs..."):
                try:
                    vector_db = create_vector_db(all_uploads)
                    st.session_state["chats"][active]["vector_db"] = vector_db
                    st.session_state["chats"][active]["pdf_pages"] = extract_all_pages_as_images(all_uploads)
                    st.success("✅ PDFs processed.")
                except Exception as e:
                    st.error(f"Error: {e}")

        # PDF Preview with zoom
        if active and st.session_state["chats"][active].get("pdf_pages"):
            with st.expander("📑 View PDFs"):
                zoom = st.slider("Zoom", 100, 1000, 700, 50, key=f"zoom_{active}")
                for page_info in st.session_state["chats"][active]["pdf_pages"]:
                    st.image(page_info["image"], width=zoom)
                    st.caption(f'{page_info["source"]} - Page {page_info["page"]}')

        if active and st.button("⚠️ Clear PDFs for this chat"):
            st.session_state["chats"][active]["vector_db"] = None
            st.session_state["chats"][active]["file_uploads"] = []
            st.session_state["chats"][active]["pdf_pages"] = []
            st.success("Cleared.")

    # ---------------- Right Column ----------------
    with col2:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 Model")
        available_models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash-lite-001",
            "HuggingFaceH4/zephyr-7b-beta",
            "google/gemma-7b",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]
        selected_model = st.sidebar.selectbox("Choose model ↓", available_models, index=0)

        if ("llm" not in st.session_state) or (st.session_state.get("selected_model") != selected_model):
            with st.spinner(f"Loading {selected_model}..."):
                try:
                    if "gemini" in selected_model.lower():
                        st.session_state["llm"] = get_gemini_llm(selected_model)
                    else:
                        st.session_state["llm"] = get_hf_pipeline(selected_model)
                    st.session_state["selected_model"] = selected_model
                    st.success(f"{selected_model} loaded.")
                except Exception as e:
                    st.error(f"Failed: {e}")

        st.subheader("💬 Chat")
        if active is None:
            st.info("Open or create a chat from the sidebar.")
            return

        messages = st.session_state["chats"][active].get("messages", [])
        message_container = st.container()
        for message in messages:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask about the PDFs...")
        if prompt:
            msg_user = {"role": "user", "content": prompt, "ts": datetime.now().isoformat()}
            st.session_state["chats"][active]["messages"].append(msg_user)
            with message_container.chat_message("user", avatar="😎"):
                st.markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    vector_db = st.session_state["chats"][active].get("vector_db")
                    llm = st.session_state.get("llm")
                    try:
                        response = process_question(prompt, vector_db, llm)
                    except Exception as e:
                        response = f"Error: {e}"
                    st.markdown(response)
                    msg_assistant = {"role": "assistant", "content": response, "ts": datetime.now().isoformat()}
                    st.session_state["chats"][active]["messages"].append(msg_assistant)

            # Rename chat if default
            if active.startswith("Chat - "):
                first_msg = prompt.strip()
                short = (first_msg[:40] + "...") if len(first_msg) > 40 else first_msg
                new_name = f"{short} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                st.session_state["chats"][new_name] = st.session_state["chats"].pop(active)
                st.session_state["active_chat"] = new_name


if __name__ == "__main__":
    main()
