import streamlit as st
import os
import tempfile
import gc
import base64
import io
import time
import uuid
import hashlib
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from pypdf import PdfReader, PdfWriter
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task, LLM
from custom_tool import DocumentSearchTool, FireCrawlWebSearchTool

load_dotenv()

@st.cache_resource
def load_llm():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = LLM(
        model="ollama/llama3.2:3b",
        base_url=base_url
    )
    return llm

def check_ollama_connection() -> tuple[bool, str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    health_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urlopen(health_url, timeout=5) as response:
            if response.status >= 400:
                return False, f"Ollama returned status {response.status}."
    except HTTPError as exc:
        return False, f"Ollama returned status {exc.code}."
    except URLError:
        return False, "Unable to connect to Ollama. Ensure it is running and reachable."
    return True, ""

# ===========================
#   Define Agents & Tasks
# ===========================
def create_agents_and_tasks():
    """Creates a Crew with a single response synthesizer agent."""
    response_synthesizer_agent = Agent(
        role="Response synthesizer agent for the user query: {query}",
        goal=(
            "Synthesize the retrieved information into a concise and coherent response "
            "based on the user query: {query}. If you are not able to retrieve the "
            'information then respond with "I\'m sorry, I couldn\'t find the information '
            'you\'re looking for."'
        ),
        backstory=(
            "You're a skilled communicator with a knack for turning "
            "complex information into clear and concise responses."
        ),
        verbose=True,
        llm=load_llm()
    )

    response_task = Task(
        description=(
            "Synthesize the final response for the user query: {query} using the "
            "retrieved context below.\n\n"
            "Context:\n{context}"
        ),
        expected_output=(
            "A concise and coherent response based on the retrieved information "
            "from the right source for the user query: {query}. If you are not "
            "able to retrieve the information, then respond with: "
            '"I\'m sorry, I couldn\'t find the information you\'re looking for."'
        ),
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[response_synthesizer_agent],
        tasks=[response_task],
        process=Process.sequential,  # or Process.hierarchical
        verbose=True
    )
    return crew

def retrieve_context(query: str, pdf_tool: DocumentSearchTool | None) -> str:
    """Retrieve context from PDF tool and web search (when a URL is provided)."""
    context_parts = []
    has_url = "http://" in query or "https://" in query

    if pdf_tool is not None:
        try:
            pdf_result = pdf_tool._run(query)
        except Exception as exc:
            # Structured fallback when the tool raises unexpectedly
            pdf_result = None
        # Only add PDF context when the search succeeded and has content
        if pdf_result and getattr(pdf_result, "ok", False) and getattr(pdf_result, "text", "").strip():
            context_parts.append(f"PDF context:\n{pdf_result.text}")

    if (has_url or not context_parts) and FireCrawlWebSearchTool.is_enabled():
        web_result = FireCrawlWebSearchTool()._run(query)
        if isinstance(web_result, str) and web_result.strip():
            context_parts.append(f"Web context:\n{web_result}")

    if not context_parts:
        return "No context available."

    return "\n\n".join(context_parts)

# ===========================
#   Streamlit Setup
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None  # Store the DocumentSearchTool

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None  # Persisted path to uploaded PDF

if "pdf_fingerprint" not in st.session_state:
    st.session_state.pdf_fingerprint = None  # Track current PDF identity

if "crew" not in st.session_state:
    st.session_state.crew = None      # Store the Crew object

def _cleanup_temp_pdf():
    pdf_path = st.session_state.get("pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except OSError:
            pass
    st.session_state.pdf_path = None
    st.session_state.pdf_tool = None
    st.session_state.pdf_fingerprint = None

def reset_chat():
    st.session_state.messages = []
    _cleanup_temp_pdf()
    gc.collect()

def _preview_first_pages(file_bytes: bytes, max_pages: int = 3) -> bytes:
    """Return a PDF containing only the first up to `max_pages` pages.

    Falls back to original bytes on parse/write errors.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        writer = PdfWriter()
        total = len(reader.pages)
        limit = min(max_pages, total)
        for i in range(limit):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return file_bytes

def display_pdf(file_bytes: bytes, file_name: str):
    """Displays a preview of the first up to 3 pages of the PDF."""
    preview_bytes = _preview_first_pages(file_bytes, max_pages=3)
    base64_pdf = base64.b64encode(preview_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name} (first up to 3 pages)")
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    # ===========================
    #   Streamlit Setup
    # ===========================
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Chat history

    if "pdf_tool" not in st.session_state:
        st.session_state.pdf_tool = None  # Store the DocumentSearchTool

    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None  # Persisted path to uploaded PDF

    if "crew" not in st.session_state:
        st.session_state.crew = None      # Store the Crew object

    # ===========================
    #   Sidebar
    # ===========================
    with st.sidebar:
        st.header("Add Your PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            uploaded_fingerprint = hashlib.sha1(uploaded_bytes).hexdigest()
            # If there's a new file and we haven't set pdf_tool yet...
            if st.session_state.pdf_tool is None or uploaded_fingerprint != st.session_state.pdf_fingerprint:
                if st.session_state.pdf_tool is not None:
                    _cleanup_temp_pdf()
                temp_dir = tempfile.gettempdir()
                safe_name = f"rag-upload-{uuid.uuid4().hex}-{uploaded_file.name}"
                temp_file_path = os.path.join(temp_dir, safe_name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_bytes)

                st.session_state.pdf_path = temp_file_path
                st.session_state.pdf_fingerprint = uploaded_fingerprint

                with st.spinner("Indexing PDF... Please wait..."):
                    st.session_state.pdf_tool = DocumentSearchTool(file_path=temp_file_path)
                
                st.success("PDF indexed! Ready to chat.")

            # Optionally display the PDF in the sidebar
            display_pdf(uploaded_file.getvalue(), uploaded_file.name)

        st.button("Clear Chat", on_click=reset_chat)

        if not FireCrawlWebSearchTool.is_enabled():
            st.warning("Web search is disabled because the FireCrawl API key is missing or uses the default placeholder.")

    # ===========================
    #   Main Chat Interface
    # ===========================
    st.markdown("""
        # Agentic RAG powered by <img src="data:image/png;base64,{}" width="120" height="120" style="vertical-align: -3px;">
    """.format(base64.b64encode(open("static/ollama.png", "rb").read()).decode()), unsafe_allow_html=True)

    ollama_ok, ollama_error = check_ollama_connection()
    if not ollama_ok:
        st.error(f"Ollama connection error: {ollama_error}")
        st.stop()  # Stop the app if Ollama is not reachable

    # Render existing conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask a question about your PDF...")

    if prompt:
        # 1. Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Build or reuse the Crew (only once after PDF is loaded)
        if st.session_state.crew is None:
            st.session_state.crew = create_agents_and_tasks()

        # 3. Get the response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Get the complete response first
            with st.spinner("Thinking..."):
                retrieved_context = retrieve_context(prompt, st.session_state.pdf_tool)
                inputs = {"query": prompt, "context": retrieved_context}
                result = st.session_state.crew.kickoff(inputs=inputs).raw
            
            # Split by lines first to preserve code blocks and other markdown
            lines = result.split('\n')
            for i, line in enumerate(lines):
                full_response += line
                if i < len(lines) - 1:  # Don't add newline to the last line
                    full_response += '\n'
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.15)  # Adjust the speed as needed
            
            # Show the final response without the cursor
            message_placeholder.markdown(full_response)

        # 4. Save assistant's message to session
        st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _cleanup_temp_pdf()
        raise
