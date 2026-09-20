from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    PROMPT_TEMPLATE = (
        "Answer the question using ONLY the context below.\n"
        "If the context does not contain the answer, say you do not know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve the most relevant chunks.
        results = self.store.search(question, top_k=top_k)

        if not results:
            return "I don't know — no relevant context was found in the knowledge base."

        # 2. Build the context block, numbering sources so answers stay traceable.
        context = "\n\n".join(
            f"[{index}] (score={result['score']:.3f}) {result['content']}"
            for index, result in enumerate(results, start=1)
        )
        prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)

        # 3. Hand the grounded prompt to the LLM.
        return self.llm_fn(prompt)
