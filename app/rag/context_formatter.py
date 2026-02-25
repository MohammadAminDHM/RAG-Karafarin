from typing import Dict, List


def format_context_blocks(results: List[Dict], max_context_chars: int = 2400) -> str:
    """
    Convert retrieved chunks into a compact context string.
    """
    if not results:
        return ""

    blocks = []
    used = 0
    for i, r in enumerate(results, start=1):
        text = (r.get("text") or "").strip()
        score = r.get("score", 0.0)
        chunk_id = r.get("chunk_id", f"chunk-{i}")
        block = f"[{i}] ({chunk_id}, score={score:.4f})\n{text}"
        if used + len(block) > max_context_chars and blocks:
            break
        blocks.append(block)
        used += len(block) + 2

    return "\n\n".join(blocks)
