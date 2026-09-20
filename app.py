import streamlit as st
import streamlit.components.v1 as components
from src.pdf_processor import extract_pdf_pages, render_pdf_page_image
from src.text_splitter import split_pages_into_chunks
from src.vector_store import get_chroma_collection, store_chunks, query_vector_store
from src.answer_generator import generate_rag_answer, generate_quick_quiz, generate_knowledge_graph
from src.text_enhancer import correct_text, predict_next_words
from src.graph_visualizer import render_interactive_graph

st.set_page_config(
    page_title="AI Study Assistant Pro",
    page_icon="📚",
    layout="wide"
)
# --- Cyberpunk Tech & High-Glow Glassmorphism Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Ambient Animated Mesh Background with Glowing Neon Gradients */
    .stApp, 
    div[data-testid="stAppViewContainer"], 
    div[data-testid="stAppViewBlockContainer"],
    section.main {
        background-color: #05070e !important;
        background-image: 
            radial-gradient(at 10% 15%, rgba(0, 242, 254, 0.18) 0px, transparent 50%),
            radial-gradient(at 88% 18%, rgba(79, 172, 254, 0.22) 0px, transparent 50%),
            radial-gradient(at 50% 85%, rgba(138, 43, 226, 0.18) 0px, transparent 55%),
            radial-gradient(at 85% 85%, rgba(0, 255, 170, 0.15) 0px, transparent 45%),
            linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%, 35px 35px, 35px 35px !important;
        background-attachment: fixed !important;
        color: #f1f5f9 !important;
    }

    /* Seamless Glass Header */
    header[data-testid="stHeader"] {
        background: rgba(5, 7, 14, 0.4) !important;
        backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* High-Gloss Translucent Sidebar + Ambient Neon Mesh */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebarUserContent"] {
        background-color: #060913 !important;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(0, 242, 254, 0.16) 0px, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(138, 43, 226, 0.14) 0px, transparent 45%),
            linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 30px 30px, 30px 30px !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(0, 242, 254, 0.25) !important;
        box-shadow: 10px 0 35px rgba(0, 0, 0, 0.65) !important;
    }

    /* Sidebar Headings Glowing Accent */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
        text-shadow: 0 0 14px rgba(0, 242, 254, 0.4) !important;
    }

    /* Sidebar Dividers (Glowing Neon Line) */
    section[data-testid="stSidebar"] hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0, 242, 254, 0.5), transparent) !important;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.3) !important;
        margin: 1.2rem 0 !important;
    }

    /* File Uploader Container inside Sidebar */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
        background: rgba(13, 19, 36, 0.75) !important;
        border: 1px dashed rgba(0, 242, 254, 0.4) !important;
        border-radius: 14px !important;
        padding: 0.75rem !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: inset 0 0 15px rgba(0, 242, 254, 0.05) !important;
    }

    /* Dropdown / Selectbox inside Sidebar */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: rgba(13, 19, 36, 0.85) !important;
        border: 1px solid rgba(0, 242, 254, 0.35) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }

    /* Elevated Frosted Glass Cards (Chat Messages) */
    div[data-testid="stChatMessage"] {
        background: rgba(13, 19, 36, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    div[data-testid="stChatMessage"]:hover {
        border-color: rgba(0, 242, 254, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* Cyan Aura on User Messages */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.12) 0%, rgba(0, 242, 254, 0.05) 100%) !important;
        border-left: 4px solid #00f2fe !important;
    }

    /* Neon Emerald Aura on Assistant Messages */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, rgba(0, 255, 170, 0.12) 0%, rgba(16, 185, 129, 0.05) 100%) !important;
        border-left: 4px solid #00ffaa !important;
    }

    /* Action Buttons: Glowing Gradient Pill */
    div.stButton > button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #00ffaa 100%) !important;
        background-size: 200% auto !important;
        color: #030816 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.4rem !important;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.35) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        background-position: right center !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(0, 255, 170, 0.55) !important;
        color: #000000 !important;
    }

    /* Floating Capsule Chat Input */
    div[data-testid="stChatInput"] {
        background: rgba(10, 16, 31, 0.8) !important;
        border: 1px solid rgba(0, 242, 254, 0.5) !important;
        border-radius: 18px !important;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.25) !important;
        backdrop-filter: blur(14px) !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #00ffaa !important;
        box-shadow: 0 0 30px rgba(0, 255, 170, 0.4) !important;
    }

    /* Modern Tabs with Underline Glow */
    button[data-baseweb="tab"] {
        background: transparent !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: #94a3b8 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f2fe !important;
        border-bottom: 2px solid #00f2fe !important;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5) !important;
    }

    /* Source Page Card & Graph Container Glow */
    div[data-testid="stImage"] {
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid rgba(0, 242, 254, 0.25) !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 242, 254, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)# Persistent Session States
if "collection" not in st.session_state:
    st.session_state.collection = get_chroma_collection()
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "uploaded_pdf_bytes" not in st.session_state:
    st.session_state.uploaded_pdf_bytes = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page_view" not in st.session_state:
    st.session_state.current_page_view = 1
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("# 📚 **Study Assistant Pro**")
    st.caption("Multilingual RAG with Visual Grounding & Concept Maps")
    st.divider()

    st.header("⚙️ Preferences")
    selected_language = st.selectbox(
        "Response Language",
        options=["English", "Tamil (தமிழ்)", "Hindi (हिन्दी)", "Telugu (తెలుగు)", "Kannada (ಕನ್ನಡ)"],
        index=0
    )

    st.divider()
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload Lecture Notes (PDF)", type=["pdf"])

    if uploaded_file and not st.session_state.doc_loaded:
        if st.button("⚡ Process Document", use_container_width=True):
            with st.spinner("Extracting pages & indexing vectors..."):
                pdf_bytes = uploaded_file.read()
                st.session_state.uploaded_pdf_bytes = pdf_bytes
                st.session_state.uploaded_filename = uploaded_file.name

                pages_data = extract_pdf_pages(pdf_bytes, uploaded_file.name)
                chunks = split_pages_into_chunks(pages_data)
                store_chunks(st.session_state.collection, chunks)

                st.session_state.doc_loaded = True
                st.success(f"Indexed {len(pages_data)} pages ({len(chunks)} chunks)!")

    st.divider()
    st.header("🎯 Advanced Tools")
    if st.button("🕸️ Build Concept Map", use_container_width=True, disabled=not st.session_state.doc_loaded):
        with st.spinner("Analyzing semantic relationships..."):
            key_chunks = query_vector_store(
                st.session_state.collection,
                "definitions concepts core theory relationships architecture components",
                top_k=6
            )
            st.session_state.graph_data = generate_knowledge_graph(key_chunks)
            st.rerun()

    if st.button("📝 Generate Quick Quiz", use_container_width=True, disabled=not st.session_state.doc_loaded):
        with st.spinner("Generating quiz..."):
            sample_chunks = query_vector_store(st.session_state.collection, "core concept definition principle", top_k=4)
            quiz = generate_quick_quiz(sample_chunks, language=selected_language)
            st.session_state.messages.append({"role": "assistant", "content": f"### 📝 Quick Quiz\n\n{quiz}", "page": None})
            st.rerun()

# --- Main Layout: Two Columns ---
col_chat, col_viewer = st.columns([1.1, 0.9], gap="medium")

# --- LEFT COLUMN: Chat Interface ---
with col_chat:
    st.subheader("💬 Study Dialogue")

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("page"):
                if st.button(f"🔍 Inspect Page {msg['page']} in Viewer", key=f"btn_{idx}_{msg['page']}"):
                    st.session_state.current_page_view = msg["page"]
                    st.rerun()

    user_query = st.chat_input("Ask a question about your study material...")

    if user_query:
        if not st.session_state.doc_loaded:
            st.warning("⚠️ Please upload a PDF and click 'Process Document' first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query, "page": None})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                cleaned_query = correct_text(user_query)
                if cleaned_query.lower() != user_query.lower():
                    st.info(f"💡 Corrected spelling: *{cleaned_query}*")

                # Detect user intent for mind map generation directly from chat
                graph_trigger_keywords = ["mind map", "concept map", "map out", "architecture", "relationship graph", "ontology", "flow chart"]
                is_graph_request = any(k in cleaned_query.lower() for k in graph_trigger_keywords)

                if is_graph_request:
                    with st.spinner("Extracting entities and building concept map..."):
                        key_chunks = query_vector_store(st.session_state.collection, cleaned_query, top_k=6)
                        st.session_state.graph_data = generate_knowledge_graph(key_chunks)
                        answer = "I've extracted the key entities and their semantic links. Switch to the **Interactive Concept Map** tab on the right to explore the graph!"
                        st.markdown(answer)
                        top_page = None
                else:
                    suggestions = predict_next_words(cleaned_query)
                    if suggestions:
                        st.caption("Predicted next words: " + ", ".join([f"`{w}`" for w in suggestions]))

                    with st.spinner("Retrieving notes & drafting answer..."):
                        contexts = query_vector_store(st.session_state.collection, cleaned_query, top_k=3)
                        top_page = contexts[0]["page"] if contexts else 1
                        answer = generate_rag_answer(cleaned_query, contexts, language=selected_language)

                        st.markdown(answer)
                        st.session_state.current_page_view = top_page

                        with st.expander("🔍 View Retrieved Chunks"):
                            for c_idx, c in enumerate(contexts, start=1):
                                st.caption(f"**Source {c_idx}** — Page {c['page']} ({c['document']})")
                                st.text(c["text"][:250] + "...")

            st.session_state.messages.append({"role": "assistant", "content": answer, "page": top_page})
            st.rerun()

# --- RIGHT COLUMN: Visual Grounding & Knowledge Graph ---
with col_viewer:
    tab_doc, tab_graph = st.tabs(["📄 Visual Source Page", "🕸️ Interactive Concept Map"])

    with tab_doc:
        st.subheader("Source Document Grounding")
        if st.session_state.doc_loaded and st.session_state.uploaded_pdf_bytes:
            st.caption(f"Viewing **{st.session_state.uploaded_filename}** — **Page {st.session_state.current_page_view}**")
            try:
                page_img = render_pdf_page_image(
                    st.session_state.uploaded_pdf_bytes,
                    st.session_state.current_page_view
                )
                st.image(page_img, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render page image: {e}")
        else:
            st.info("Upload a document to view real-time source page verifications here.")

    with tab_graph:
        st.subheader("Concept Mind Map")
        if st.session_state.graph_data and st.session_state.graph_data.get("nodes"):
            html_graph = render_interactive_graph(st.session_state.graph_data)
            if html_graph:
                components.html(html_graph, height=540, scrolling=False)
            else:
                st.warning("Could not construct graph representation.")
        else:
            st.info("Click '🕸️ Build Concept Map' in the sidebar or ask to map concepts in the chat to generate an interactive mind map.")
