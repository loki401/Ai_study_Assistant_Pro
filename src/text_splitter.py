def split_pages_into_chunks(pages_data, chunk_size=800, overlap=150):
    """
    Splits page-level text into smaller overlapping chunks,
    preserving document metadata and page numbers.
    """
    chunks = []
    chunk_id = 0

    for item in pages_data:
        text = item["text"]
        page_num = item["page"]
        doc_name = item["document"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "id": f"{doc_name}_p{page_num}_c{chunk_id}",
                "text": chunk_text,
                "metadata": {
                    "page": page_num,
                    "document": doc_name
                }
            })
            chunk_id += 1
            start += (chunk_size - overlap)

    return chunks