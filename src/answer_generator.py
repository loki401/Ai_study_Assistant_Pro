import os
import json
from dotenv import load_dotenv, find_dotenv
from google import genai

# Load .env
load_dotenv(find_dotenv(usecwd=True), override=True)
raw_key = os.getenv("GEMINI_API_KEY")

if not raw_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

# Strip any accidental terminal spaces
api_key = raw_key.strip()
client = genai.Client(api_key=api_key)

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
        model="gemini-3.6-flash",
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
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text
import json

def generate_knowledge_graph(retrieved_contexts: list) -> dict:
    """
    Extracts key academic entities and relationships into an edge list JSON
    for network visualization.
    """
    if not retrieved_contexts:
        return {"nodes": [], "edges": []}

    context_str = "\n\n".join([f"Page {c['page']}: {c['text']}" for c in retrieved_contexts[:5]])

    prompt = f"""Analyze these study notes and identify 6 to 10 core concepts/entities and their relationships.
Return STRICTLY valid JSON with no markdown backticks, no preamble, and no extra text.
The JSON format must strictly be:
{{
  "nodes": ["ConceptA", "ConceptB", "ConceptC"],
  "edges": [
    {{"source": "ConceptA", "target": "ConceptB", "label": "causes"}},
    {{"source": "ConceptB", "target": "ConceptC", "label": "part of"}}
  ]
}}

Context:
{context_str}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    
    clean_text = response.text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_text)
    except Exception:
        return {"nodes": [], "edges": []}
