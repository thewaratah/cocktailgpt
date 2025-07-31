import fitz  # PyMuPDF
import re
from io import BytesIO

def extract_text_from_pdf(source):
    """
    Accepts either a file path (str) or a BytesIO object.
    Returns the full extracted text from the PDF.
    """
    if isinstance(source, str):
        doc = fitz.open(source)
    elif isinstance(source, BytesIO):
        doc = fitz.open(stream=source, filetype="pdf")
    else:
        raise ValueError("source must be a file path or BytesIO")

    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\S\r\n]{2,}', ' ', text)
    return text.strip()

def chunk_text(text, max_tokens=500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current + para) < max_tokens * 4:  # Approx. 4 chars/token
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def format_response_with_citations(answer: str, results: dict) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    citations = []

    for i, meta in enumerate(metas):
        if isinstance(meta, dict):
            source = meta.get("source") or meta.get("path") or "Unknown"
        elif isinstance(meta, str):
            source = meta
        else:
            source = "Unknown"
        citations.append(f"[{i+1}] {source}")

    if citations:
        answer += "\n\n📚 Sources:\n" + "\n".join(citations)
    else:
        answer += "\n\n📚 Sources:\n[No chunk citations found.]"

    return answer
