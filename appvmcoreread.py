import os
import tarfile
import shutil
import gradio as gr
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from fpdf import FPDF
import subprocess

# Directories
UPLOAD_DIR = "uploaded_reports"
EXTRACT_DIR = "extracted_reports"
PDF_DIR = "sos_pdf"
VECTOR_DIR = "vector_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

qa = None  # Global QA chain

# -------------------- VMCORE Analysis --------------------
def analyze_vmcore(extract_path):
    summary = ""
    for root, dirs, files in os.walk(extract_path):
        for fname in files:
            if "vmcore" in fname:
                vmcore_path = os.path.join(root, fname)
                try:
                    result = subprocess.run(
                        ["crash", "--batch", "--stdio", vmcore_path],
                        capture_output=True, text=True, timeout=30
                    )
                    summary = result.stdout[:5000]
                except Exception as e:
                    summary = f"⚠️ VMCORE analysis failed: {str(e)}"
    return summary or "⚠️ No vmcore found for analysis."

# -------------------- PDF Generation --------------------
def generate_pdf(extract_path, vmcore_summary, pdf_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add Unicode font
    pdf.add_page()
    pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", '', 10)

    # VMCORE summary
    pdf.multi_cell(0, 10, "VMCORE Crash Summary\n\n" + vmcore_summary)

    # Logs & configs
    for root, dirs, files in os.walk(extract_path):
        for fname in files:
            if fname.endswith((".log", ".txt", ".conf", ".out", ".err", "dmesg")):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", errors="ignore") as f:
                        content = f.read()
                        pdf.add_page()
                        pdf.multi_cell(0, 10, f"File: {fpath}\n\n{content[:5000]}")
                except:
                    continue
    pdf.output(pdf_path)

# -------------------- SOS Report Processing --------------------
def process_sosreport(sos_file):
    global qa
    status_msgs = []

    # Save uploaded file (bytes)
    filename = "uploaded_sosreport.tar.gz"
    upload_path = os.path.join(UPLOAD_DIR, filename)
    with open(upload_path, "wb") as f:
        f.write(sos_file)

    # Cleanup previous extraction
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Extract tarball
    extract_path = os.path.join(EXTRACT_DIR, "sos_extracted")
    os.makedirs(extract_path, exist_ok=True)
    try:
        with tarfile.open(upload_path, "r:*") as tar:
            tar.extractall(extract_path)
        status_msgs.append(f"✅ Extracted SOS report to {extract_path}")
    except Exception as e:
        return f"⚠️ Failed to extract sosreport: {str(e)}", None

    # VMCORE analysis
    vmcore_summary = analyze_vmcore(extract_path)
    status_msgs.append("✅ VMCORE analysis done")

    # PDF generation
    pdf_path = os.path.join(PDF_DIR, "sosreport_summary.pdf")
    generate_pdf(extract_path, vmcore_summary, pdf_path)
    status_msgs.append(f"✅ PDF generated: {pdf_path}")

    # Prepare QA chain
    docs = [vmcore_summary]
    for root, dirs, files in os.walk(extract_path):
        for fname in files:
            if fname.endswith((".log", ".txt", ".conf", ".out", ".err", "dmesg")):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", errors="ignore") as f:
                        docs.append(f.read())
                except:
                    continue

    if not docs:
        return "⚠️ No readable files found.", pdf_path

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_DIR)
    retriever = vectordb.as_retriever(search_kwargs={"k":5})

    llm = OllamaLLM(model="llama3")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    status_msgs.append("✅ QA chain ready! Ask questions now.")

    return "\n".join(status_msgs), pdf_path

# -------------------- QA --------------------
def ask_question(question):
    global qa
    if qa is None:
        return "⚠️ Please upload & process a SOS report first."
    try:
        return qa.run(question)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# -------------------- Gradio UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("## SOS Report + VMCORE Analyzer with Crash Detection (Offline)")
    with gr.Row():
        with gr.Column():
            sos_file = gr.File(label="Upload SOS report (.tar.xz/.tar.gz)", type="binary")
            upload_btn = gr.Button("Process SOS report")
            status = gr.Textbox(label="Status", lines=20)
            pdf_download = gr.File(label="Download PDF", file_types=[".pdf"])
        with gr.Column():
            question = gr.Textbox(label="Ask a Question")
            ask_btn = gr.Button("Ask")
            answer = gr.Textbox(label="Answer", lines=10)

    upload_btn.click(process_sosreport, inputs=sos_file, outputs=[status, pdf_download])
    ask_btn.click(ask_question, inputs=question, outputs=answer)

demo.launch(server_name="0.0.0.0", server_port=7860)
