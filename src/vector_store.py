import chromadb
from chromadb.utils import embedding_functions

# Use Chroma's lightweight default ONNX MiniLM embedding model
default_ef = embedding_functions.DefaultEmbeddingFunction()

def get_chroma_collection(collection_name="study_assistant"):
    """Creates or accesses an in-memory Chroma collection."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=default_ef
    )
    return collection

def store_chunks(collection, chunks: list):
    """Upserts chunk text and page metadata into the vector collection."""
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

def query_vector_store(collection, query_text: str, top_k=3):
    """
    Performs similarity search and returns the top_k most relevant chunks
    with their page numbers.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )

    retrieved = []
    if results and results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved.append({
                "text": doc,
                "page": meta["page"],
                "document": meta["document"]
            })
    return retrieved