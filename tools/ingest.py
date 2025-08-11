# Document ingestion skeleton
# - chunking strategies
# - metadata extraction
# - embedding pipeline (stubbed)
def chunk_text(text: str, size: int = 800, overlap: int = 120):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks
