from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Minimal Persian/English stopwords (keep small & safe)
STOPWORDS = {
    "و","یا","به","از","در","را","با","برای","این","آن","یک","که","تا","هم","اما","اگر","پس","می","شود","شده","کرد","کردن","کرده",
    "the","a","an","to","of","in","on","and","or","is","are","was","were","be","for","with","as","it","this","that"
}

META_PATTERNS = [
    r"این\s*فایل",
    r"درباره\s*چیه",
    r"محتوا(ی)?\s*فایل",
    r"jsonl",
    r"دیتاست",
    r"dataset",
    r"چه\s*چیزهایی\s*داخل",
    r"چه\s*موضوعاتی",
    r"خلاصه",
    r"summary",
    r"ساختار",
    r"structure",
]


def is_meta_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    for pat in META_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return True
    return False


def _extract_question(text: str) -> str:
    """
    Extract the 'Question:' line from a QA-formatted text block.
    """
    if not text:
        return ""
    # common patterns
    m = re.search(r"^\s*question\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^\s*prompt\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()

    # fallback: first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def compute_corpus_summary(
    faiss_metadata_path: str,
    index_state_path: Optional[str] = None,
    max_scan_items: int = 4000,
    top_terms: int = 12,
    sample_questions: int = 8,
) -> Dict[str, Any]:
    """
    Compute a lightweight dataset summary from FAISS metadata JSON.
    Works for JSONL QA datasets.
    """
    meta_path = Path(faiss_metadata_path)
    if not meta_path.exists():
        return {
            "ok": False,
            "error": "faiss_metadata_missing",
            "faiss_metadata_path": str(meta_path),
        }

    items = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        return {
            "ok": False,
            "error": "faiss_metadata_empty",
            "faiss_metadata_path": str(meta_path),
        }

    # Optionally read index_state
    index_state = {}
    if index_state_path:
        p = Path(index_state_path)
        if p.exists():
            try:
                index_state = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                index_state = {}

    # Scan a subset for speed
    scan = items[:max_scan_items]

    doc_ids = set()
    record_indices = set()
    term_counter: Counter = Counter()
    examples: List[str] = []

    for it in scan:
        if not isinstance(it, dict):
            continue
        doc_id = it.get("doc_id")
        if isinstance(doc_id, str):
            doc_ids.add(doc_id)
        ri = it.get("record_index")
        if isinstance(ri, int):
            record_indices.add(ri)

        text = it.get("text", "") or ""
        q = _extract_question(text)
        if q:
            # collect examples
            if len(examples) < sample_questions:
                examples.append(q)
            # collect terms
            term_counter.update(_tokenize(q))

    top = [t for t, _ in term_counter.most_common(top_terms)]

    # Try to guess domain label from top terms
    domain_hint = ""
    joined = " ".join(top)
    if "بانک" in joined or "کارآفرین" in joined or "های" in joined or "bank" in joined:
        domain_hint = "پشتیبانی/FAQ بانکی (احتمالاً اپلیکیشن یا خدمات بانک)"
    elif "رمز" in joined or "پویا" in joined or "شاهکار" in joined:
        domain_hint = "موضوعات پرتکرار مربوط به رمز/احراز هویت/پیامک"

    summary = {
        "ok": True,
        "total_chunks_in_metadata": len(items),
        "scanned_chunks": len(scan),
        "unique_doc_ids_scanned": len(doc_ids),
        "unique_record_indices_scanned": len(record_indices),
        "top_terms": top,
        "sample_questions": examples,
        "index_state": index_state,
        "domain_hint": domain_hint,
    }
    return summary


def format_summary_fa(summary: Dict[str, Any]) -> str:
    if not summary.get("ok"):
        return f"خلاصه دیتاست در دسترس نیست. دلیل: {summary.get('error')}"

    top_terms = summary.get("top_terms") or []
    samples = summary.get("sample_questions") or []
    domain_hint = summary.get("domain_hint") or ""

    n_chunks = summary.get("total_chunks_in_metadata", 0)
    n_docs = summary.get("unique_doc_ids_scanned", 0)
    scanned = summary.get("scanned_chunks", 0)

    lines = []
    lines.append("این فایل یک دیتاست پرسش/پاسخ (QA) است که برای پاسخ‌گویی و پشتیبانی طراحی شده.")
    if domain_hint:
        lines.append(f"برداشت سریع از موضوع: {domain_hint}")
    lines.append(f"- تعداد چانک‌های ایندکس‌شده (طبق metadata): {n_chunks}")
    lines.append(f"- تعداد نمونه بررسی‌شده برای خلاصه: {scanned}")
    if n_docs:
        lines.append(f"- تعداد رکورد/Doc مشاهده‌شده در نمونه‌ها: {n_docs}")

    if top_terms:
        lines.append("\nموضوعات پرتکرار (تقریبی):")
        lines.append("، ".join(top_terms))

    if samples:
        lines.append("\nنمونه سوال‌ها:")
        for i, s in enumerate(samples[:8], start=1):
            lines.append(f"{i}) {s}")

    lines.append("\nاگر می‌خوای دقیق‌تر کمک کنم: یک سوال مشخص از جنس مشتری بپرس (مثلاً «رمز پویا نمیاد»)، تا جواب مرتبط از دیتاست برگردونم.")
    return "\n".join(lines)
