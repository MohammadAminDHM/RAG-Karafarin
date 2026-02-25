def build_rag_prompt(query: str, context: str) -> str:
    """
    Prompt builder for future local/API LLM generation phases.
    Phase 1 prepares the prompt but uses heuristic answering.
    """
    return (
        "تو دستیار بانک کارآفرین هستی و باید به سوالات کاربر پاسخ دهی.\n\n"
        f"سوال:\n{query}\n\n"
        f"متن مرتبط با سوال:\n{context}\n\n"
        "اگر پاسخی در متن مرتبط نبود، بگو که نمی دونی."
    )
