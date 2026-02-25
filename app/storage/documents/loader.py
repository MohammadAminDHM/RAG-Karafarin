from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _record_to_text(obj: Any) -> str:
    """
    Convert a JSONL record to a text string for embedding.
    Priority:
      - text / content / body
      - question/answer or q/a
      - prompt/completion
      - fallback: JSON string
    """
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj.strip()

    if not isinstance(obj, dict):
        # fallback for numbers/lists etc.
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj).strip()

    # Common direct text fields
    for k in ("text", "content", "body", "document", "message"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # QA style
    q = obj.get("question") or obj.get("q") or obj.get("query")
    a = obj.get("answer") or obj.get("a") or obj.get("response") or obj.get("completion")
    if isinstance(q, str) and q.strip():
        if isinstance(a, str) and a.strip():
            return f"Question: {q.strip()}\nAnswer: {a.strip()}"
        return f"Question: {q.strip()}"

    # prompt/completion pair
    p = obj.get("prompt")
    c = obj.get("completion")
    if isinstance(p, str) and p.strip():
        if isinstance(c, str) and c.strip():
            return f"Prompt: {p.strip()}\nCompletion: {c.strip()}"
        return f"Prompt: {p.strip()}"

    # fallback: entire object as text
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj).strip()


def load_source_documents(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load a source file and return a list of documents:
      [{ "doc_id": "...", "text": "...", "meta": {...} }, ...]

    Supports:
      - .txt -> single document
      - .jsonl -> one document per line (record)
    """
    file_path = Path(path)
    report: Dict[str, Any] = {
        "source_path": str(file_path),
        "exists": file_path.exists(),
        "file_type": file_path.suffix.lower(),
    }

    if not file_path.exists():
        return [], report

    suffix = file_path.suffix.lower()

    if suffix == ".jsonl":
        docs: List[Dict[str, Any]] = []
        total = 0
        parsed = 0
        skipped = 0
        bad_json = 0

        for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            total += 1
            s = line.strip()
            if not s:
                skipped += 1
                continue

            try:
                obj = json.loads(s)
            except Exception:
                bad_json += 1
                continue

            text = _record_to_text(obj)
            if not text.strip():
                skipped += 1
                continue

            docs.append(
                {
                    "doc_id": f"jsonl-{parsed:06d}",
                    "text": text,
                    "meta": {
                        "record_index": parsed,
                    },
                }
            )
            parsed += 1

        report.update(
            {
                "jsonl_total_lines": total,
                "jsonl_parsed_records": parsed,
                "jsonl_skipped": skipped,
                "jsonl_bad_json": bad_json,
            }
        )
        return docs, report

    # Default: treat as plain text
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    docs = [
        {
            "doc_id": "doc-000000",
            "text": text,
            "meta": {},
        }
    ]
    report.update({"raw_chars": len(text)})
    return docs, report
