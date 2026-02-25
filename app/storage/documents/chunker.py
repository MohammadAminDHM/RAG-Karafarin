from typing import Dict, List


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[Dict]:
    """
    Character-based chunking with overlap.

    Returns list of dicts:
    {
      "chunk_id": "chunk-0000",
      "text": "...",
      "start": 0,
      "end": 900
    }
    """
    if not text:
        return []

    chunk_size = max(100, int(chunk_size))
    chunk_overlap = max(0, int(chunk_overlap))
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    chunks: List[Dict] = []
    step = chunk_size - chunk_overlap
    start = 0
    idx = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "chunk_id": f"chunk-{idx:04d}",
                    "text": chunk,
                    "start": start,
                    "end": end,
                }
            )
            idx += 1
        if end >= n:
            break
        start += step

    return chunks
