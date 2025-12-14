import os
import tarfile
import shutil
import gradio as gr
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

EXTRACT_PATH = "/tmp/sosreport"
DB_PATH = "db"
qa = None  # Global QA chain

def process_sosreport(file_path, progress=gr.Progress()):
    global qa
    status_msgs = []

    # Cleanup
    status_msgs.append("🧹 Cleaning old data...")
    progress(0, desc="Cleaning old data...")
    if os.path.exists(EXTRACT_PATH):
        shutil.rmtree(EXTRACT_PATH)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    os.makedirs(EXTRACT_PATH, exist_ok=True)

    # Extract SOS
    status_msgs.append("📦 Extracting sosreport...")
    progress(0.1, desc="Extracting sosreport...")
    with tarfile.open(file_path, "r:*") as tar:
        tar.extractall(EXTRACT_PATH)

    # Load all relevant files
    status_msgs.append("📂 Loading SOS report files...")
    progress(0.2, desc="Loading files...")
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
        docs.extend(loader.load())

    if not docs:
        return "\n".join(status_msgs + ["⚠️ No readable text files found in sosreport."])

    # Split into chunks
    status_msgs.append("✂️ Splitting files into chunks...")
    progress(0.3, desc="Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create embeddings
    status_msgs.append("🧠 Creating embeddings with HuggingFace...")
    progress(0.4, desc="Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    vectordb.persist()

    # Setup QA chain
    status_msgs.append("⚙️ Setting up QA chain with Ollama (llama3)...")
    progress(0.95, desc="Setting up QA chain...")
    llm = OllamaLLM(model="llama3")
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    status_msgs.append("✅ SOS report uploaded and indexed successfully!")
    progress(1, desc="Done ✅")
    return "\n".join(status_msgs)

# Question handler
def ask_question(question):
    global qa
    if qa is None:
        return "Please upload a SOS report first."
    try:
        return qa.run(question)
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## SOS Report Analyzer (User Commands + Errors)")
    with gr.Tab("Upload SOS report"):
        sos_file = gr.File(label="Upload sosreport (.tar.xz/.tar.gz)", type="filepath")
        upload_btn = gr.Button("Process SOS report")
        status = gr.Textbox(label="Status Log", lines=20)
        upload_btn.click(process_sosreport, inputs=sos_file, outputs=status)

    with gr.Tab("Ask Questions"):
        question = gr.Textbox(placeholder="Ask about user commands, sudo, errors...")
        answer = gr.Textbox(label="Answer", lines=10)
        ask_btn = gr.Button("Ask")
        ask_btn.click(ask_question, inputs=question, outputs=answer)

demo.launch(server_name="0.0.0.0", server_port=7860)
