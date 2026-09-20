import pymupdf as fitz

def extract_pdf_pages(file_bytes: bytes, filename: str):
    """
    Extracts text page by page from an in-memory PDF byte stream.
    Returns a list of dictionaries with text, page number, and document name.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        
        if text:
            pages_data.append({
                "page": page_num + 1,
                "document": filename,
                "text": text
            })

    doc.close()
    return pages_data

def render_pdf_page_image(file_bytes: bytes, page_number: int, zoom: float = 2.0) -> bytes:
    """
    Renders a specific PDF page to high-resolution PNG bytes.
    page_number is 1-indexed.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    # Clamp page index within document bounds
    target_page = max(0, min(page_number - 1, len(doc) - 1))
    page = doc[target_page]
    
    # Render with 2x scale for sharp text rendering
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes