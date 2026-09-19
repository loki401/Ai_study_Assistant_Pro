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

# Persistent Session States
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