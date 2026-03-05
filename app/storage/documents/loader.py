from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _pick_first_str(d: dict, keys: List[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def load_source_documents(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Supports:
      - .jsonl (each line a record) -> returns docs with {doc_id, question, answer, meta}
      - .txt -> a single doc with {text}
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

        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines):
            total += 1
            s = (line or "").strip()
            if not s:
                skipped += 1
                continue
            try:
                obj = json.loads(s)
            except Exception:
                bad_json += 1
                continue
            if not isinstance(obj, dict):
                skipped += 1
                continue

            q = _pick_first_str(obj, ["question","q","query","prompt","title","input"])
            a = _pick_first_str(obj, ["answer","a","response","completion","output"])

            # fallback: if the record has text/content, treat it as q (still fine for embedding)
            if not q:
                q = _pick_first_str(obj, ["text","content","body","message","document"])
            if not q:
                skipped += 1
                continue

            docs.append({
                "doc_id": f"jsonl-{parsed:06d}",
                "question": q,
                "answer": a,
                "meta": {"record_index": parsed},
            })
            parsed += 1

        report.update({
            "jsonl_total_lines": total,
            "jsonl_parsed_records": parsed,
            "jsonl_skipped": skipped,
            "jsonl_bad_json": bad_json,
        })
        return docs, report

    # plain text fallback
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    report.update({"raw_chars": len(text)})
    return [{
        "doc_id": "doc-000000",
        "text": text,
        "meta": {},
    }], report
