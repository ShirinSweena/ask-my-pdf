import streamlit as st
import tempfile, os
from rag import build_qa_chain

st.set_page_config(page_title="Ask My PDF", page_icon="📄", layout="wide")

# Initialize session state
if "chain" not in st.session_state:
    st.session_state.chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar
with st.sidebar:
    st.title("📂 Uploaded Files")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded:
        new_files = [f.name for f in uploaded]
        if new_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = new_files
            st.session_state.chain = None
            st.session_state.messages = []

        if st.session_state.chain is None:
            with st.spinner("Processing PDFs..."):
                pdf_paths = []
                tmp_files = []
                for f in uploaded:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(f.read())
                    tmp.close()
                    pdf_paths.append((tmp.name, f.name))
                    tmp_files.append(tmp.name)

                chain, vectorstore = build_qa_chain(pdf_paths)
                st.session_state.chain = chain
                st.session_state.vectorstore = vectorstore

                for tmp_path in tmp_files:
                    os.unlink(tmp_path)

            st.success("Ready!")

    if st.session_state.uploaded_files:
        st.markdown("### 📄 Files loaded:")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"- {fname}")

    # Export chat
    if st.button("📥 Export Chat as PDF"):
        if st.session_state.messages:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Ask My PDF - Chat History", ln=True, align="C")
            pdf.ln(5)
            pdf.set_font("Helvetica", size=11)
            for msg in st.session_state.messages:
                role = "You" if msg["role"] == "user" else "Assistant"
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 8, f"{role}:", ln=True)
                pdf.set_font("Helvetica", size=10)
                # Clean text for PDF
                text = msg["content"].encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(0, 7, text)
                pdf.ln(3)
            path = "chat_export.pdf"
            pdf.output(path)
            with open(path, "rb") as f:
                st.download_button(
                    label="📄 Download PDF",
                    data=f,
                    file_name="chat_history.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("No chat history to export!")

    if st.button("🗑️ Reset Chat"):
        st.session_state.chain = None
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.rerun()

    # System monitor
    import psutil
    st.markdown("---")
    st.markdown("### 🖥️ System Monitor")
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    st.metric("CPU Usage", f"{cpu}%")
    st.metric("RAM Usage", f"{ram.percent}%", f"{ram.used // (1024**2)} MB used")
    st.button("🔄 Refresh Stats")

# Main area
st.title("📄 Ask My PDF")

if not st.session_state.uploaded_files:
    st.info("👈 Upload one or more PDFs from the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your PDF(s)..."):
    if st.session_state.chain is None:
        st.warning("Please upload a PDF first!")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": "default"}}
                )
                answer = result["answer"]

                sources = result.get("context", [])
                if sources:
                    citations = []
                    for doc in sources:
                        page = doc.metadata.get("page", None)
                        filename = doc.metadata.get("filename", "unknown")
                        if page is not None:
                            citations.append(f"{filename} (p.{page + 1})")
                    citations = list(set(citations))
                    if citations:
                        answer += f"\n\n📄 *Sources: {', '.join(sorted(citations))}*"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})