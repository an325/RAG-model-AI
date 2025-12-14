import os
import tarfile
import shutil
import gradio as gr
import re
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.schema import Document

EXTRACT_PATH = "/tmp/sosreport"
DB_PATH = "db"
qa = None  # Global QA chain

# Patterns to detect errors/warnings in logs
ERROR_PATTERNS = [
    re.compile(r"error", re.I),
    re.compile(r"failed", re.I),
    re.compile(r"warn", re.I),
    re.compile(r"critical", re.I)
]

def detect_errors_in_text(text):
    """Return True if any error/warning pattern matches"""
    return any(p.search(text) for p in ERROR_PATTERNS)

def process_sosreport(file_path, progress=gr.Progress()):
    global qa
    status_msgs = []

    # Clean old DB and extracted files
    status_msgs.append("🧹 Cleaning old data...")
    progress(0, desc="Cleaning old data...")
    if os.path.exists(EXTRACT_PATH):
        shutil.rmtree(EXTRACT_PATH)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    os.makedirs(EXTRACT_PATH, exist_ok=True)

    # Step 1: Extract sosreport
    status_msgs.append("📦 Extracting sosreport...")
    progress(0.1, desc="Extracting sosreport...")
    with tarfile.open(file_path, "r:*") as tar:
        tar.extractall(EXTRACT_PATH)

    # Step 2: Load sosreport files
    status_msgs.append("📂 Loading sosreport files...")
    progress(0.2, desc="Loading sosreport files...")
    patterns = [
        "**/*.log",
        "**/*.txt",
        "**/*.conf",
        "**/*.out",
        "**/*.err",
        "sos_commands/**/*",
        "etc/**/*"
    ]
    docs = []
    for pattern in patterns:
        loader = DirectoryLoader(EXTRACT_PATH, glob=pattern, loader_cls=TextLoader, silent_errors=True)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            # Check if chunk has error/warning and mark metadata
            if detect_errors_in_text(doc.page_content):
                doc.metadata["error_warning"] = True
            else:
                doc.metadata["error_warning"] = False
            docs.append(doc)

    if not docs:
        return "\n".join(status_msgs + ["⚠️ No readable text files found in sosreport."])

    # Step 3: Split into chunks
    status_msgs.append("✂️ Splitting files into chunks...")
    progress(0.3, desc="Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Step 4: Create embeddings with HuggingFace
    status_msgs.append("🧠 Creating embeddings with HuggingFace...")
    progress(0.4, desc="Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    vectordb.persist()

    # Step 5: Setup QA chain
    status_msgs.append("⚙️ Setting up QA chain with Ollama (llama3)...")
    progress(0.95, desc="Setting up QA chain...")
    llm = OllamaLLM(model="llama3")
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    status_msgs.append("✅ SOS report uploaded, errors detected, and QA ready!")
    progress(1, desc="Done ✅")

    return "\n".join(status_msgs)


def ask_question(question):
    global qa
    if qa is None:
        return "⚠️ Please upload and process a SOS report first."
    try:
        return qa.run(question)
    except Exception as e:
        return f"Error while processing question: {str(e)}"


# ---------- Gradio WebUI ----------
with gr.Blocks() as demo:
    gr.Markdown("## Automatic SOS Report Error Detection + QA (HuggingFace + Ollama)")

    with gr.Tab("Upload SOS Report"):
        sos_file = gr.File(label="Upload sosreport (.tar.xz or .tar.gz)", type="filepath")
        output = gr.Textbox(label="Status Log", lines=20)
        sos_file.upload(process_sosreport, inputs=sos_file, outputs=output)

    with gr.Tab("Ask Questions"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Ask about errors, warnings or system info...")
        clear = gr.Button("Clear Chat")
        def user_query(user_message, chat_history):
            response = ask_question(user_message)
            chat_history.append((user_message, response))
            return "", chat_history
        msg.submit(user_query, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860)
