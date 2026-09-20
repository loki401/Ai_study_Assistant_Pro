import os
import json
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

# Attempt loading from local .env if available
load_dotenv(find_dotenv(usecwd=True), override=True)

# 1. Retrieve the API key safely across both local and Streamlit Cloud environments
raw_key = os.getenv("GEMINI_API_KEY")

if not raw_key:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            raw_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

if not raw_key:
    raise ValueError("GEMINI_API_KEY not found! Configure it in Streamlit Cloud Secrets or your local .env.")

api_key = str(raw_key).strip()
client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.6-flash"

def generate_rag_answer(question: str, retrieved_contexts: list, language: str = "English") -> str:
    """Answers a question grounded strictly in the retrieved excerpts."""
    if not retrieved_contexts:
        return "No relevant information found in the uploaded document."

    context_blocks = []
    for c in retrieved_contexts:
        context_blocks.append(f"[Document: {c['document']} | Page {c['page']}]:\n{c['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are an academic study assistant. Answer the user's question using ONLY the provided document context excerpts.
Rules:
1. Respond completely in {language}.
2. If the answer cannot be determined from the context, respond strictly with: "I cannot find the answer in the provided notes."
3. Cite the exact page number(s) where the information was located.

Context:
{context_str}

Question:
{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text

def generate_quick_quiz(retrieved_contexts: list, language: str = "English") -> str:
    """Generates 3 multiple-choice questions from the context."""
    if not retrieved_contexts:
        return "Upload and process a document before generating a quiz."

    context_str = "\n\n".join([f"Page {c['page']}: {c['text']}" for c in retrieved_contexts[:4]])

    prompt = f"""Generate a 3-question multiple-choice study quiz based strictly on the provided text excerpts.
Rules:
1. Write the entire quiz, options, and explanations in {language}.
2. Provide 4 options (A, B, C, D) for each question.
3. Include the correct answer key and a 1-sentence explanation citing the source page number.

Excerpts:
{context_str}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text

def generate_knowledge_graph(retrieved_contexts: list) -> dict:
    """
    Extracts key academic entities and relationships into an edge list JSON
    for network visualization using Gemini's native JSON output mode.
    """
    if not retrieved_contexts:
        return {"nodes": [], "edges": []}

    context_str = "\n\n".join([f"Page {c['page']}: {c['text']}" for c in retrieved_contexts[:6]])

    prompt = f"""You are an academic knowledge graph extractor.
Analyze these study notes and identify 6 to 12 key concepts and the direct semantic relationships between them.

Schema format required:
{{
  "nodes": ["Concept1", "Concept2", "Concept3"],
  "edges": [
    {{"source": "Concept1", "target": "Concept2", "label": "leads to"}},
    {{"source": "Concept2", "target": "Concept3", "label": "component of"}}
  ]
}}

Context:
{context_str}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        if isinstance(data, dict) and "nodes" in data and "edges" in data:
            return data
    except Exception as err:
        print(f"Graph JSON parsing failed: {err}")

    return {"nodes": [], "edges": []}