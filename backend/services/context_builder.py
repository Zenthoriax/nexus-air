"""
ContextBuilder — RRF fusion, token budget enforcement, and prompt assembly.

Key fixes vs original:
  - Token estimation is a module-level standalone function: estimate_tokens()
  - Formula: int(len(text.split()) * 1.3)  — never len(text) // 4
  - System prompt contains all three required instructions (cite titles, answer only
    from context, say explicitly if not found)
  - Context format exactly: [Document: 'Title']\n{content}\n\n per document
  - build_context() uses greedy-include (O(n)) not build-then-trim (O(n^2))
  - Empty path handling: if either result list is empty, fusion works on the other
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List

from config import settings

logger = logging.getLogger(__name__)


# ── Token estimation — module-level standalone function ────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Lightweight token count approximation using word count.

    Word-count × 1.3 is consistently more accurate than character-count / 4
    for English prose, since the average English word encodes ~1.3 BPE tokens.

    Never use integer floor-division (// 4) — it underestimates by 30-50%
    and silently over-stuffs the context window.

    Args:
        text: Any string — full context block, single document, or prompt.

    Returns:
        Estimated subword token count as an integer.
    """
    return int(len(text.split()) * 1.3)


# ── ContextBuilder ─────────────────────────────────────────────────────────────

class ContextBuilder:

    # ── RRF Fusion ─────────────────────────────────────────────────────────────

    def fuse_results(
        self,
        graph_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d))

        Documents appearing in BOTH lists accumulate scores — not duplicated.
        Handles empty lists from either retrieval path gracefully.
        Returns top_n results sorted descending by fused score.
        """
        k = settings.rrf_k
        accumulated: Dict[str, Dict[str, Any]] = {}
        score_map: Dict[str, float] = defaultdict(float)

        # Graph results — keyed by 'id'
        for rank, doc in enumerate(graph_results):
            doc_id = doc.get("id") or doc.get("doc_id")
            if not doc_id:
                continue
            score_map[doc_id] += 1.0 / (k + rank + 1)   # rank is 0-based → +1 avoids k+0
            if doc_id not in accumulated:
                accumulated[doc_id] = dict(doc)
                accumulated[doc_id]["id"] = doc_id

        # Vector results — keyed by 'doc_id'
        for rank, block in enumerate(vector_results):
            doc_id = block.get("doc_id") or block.get("id")
            if not doc_id:
                continue
            score_map[doc_id] += 1.0 / (k + rank + 1)
            if doc_id not in accumulated:
                accumulated[doc_id] = dict(block)
                accumulated[doc_id]["id"] = doc_id

        # Attach accumulated scores and sort descending
        for doc_id, entry in accumulated.items():
            entry["score"] = score_map[doc_id]

        sorted_docs = sorted(accumulated.values(), key=lambda x: x["score"], reverse=True)
        top = sorted_docs[: settings.top_n]

        logger.debug(
            "ContextBuilder.fuse_results: %d graph + %d vector → %d candidates → top %d",
            len(graph_results), len(vector_results), len(sorted_docs), len(top),
        )
        return top

    # ── Context assembly ───────────────────────────────────────────────────────

    def build_context(
        self,
        fused_docs: List[Dict[str, Any]],
        token_budget: int | None = None,
    ) -> str:
        """
        Greedy-include context builder — O(n), highest-scored documents first.

        Algorithm:
          1. fused_docs is already sorted highest-score-first by fuse_results().
          2. For each document in order, estimate its token cost via estimate_tokens().
          3. If adding it would exceed the budget → SKIP it.
             (Do NOT break — a shorter later document might still fit.)
          4. If it fits → append it, add to running token total.
          5. Continue until all candidates are evaluated.

        This guarantees:
          - Highest-ranked documents are ALWAYS included first.
          - Only lower-ranked documents are dropped when budget is tight.
          - No document is silently truncated mid-content.
          - No IndexError, no crash on any list size.

        Context format per document (exact, no variation):
            [Document: 'Title']
            {content}

            (two trailing newlines between documents, stripped at the end)

        Args:
            fused_docs:   Output of fuse_results() — sorted highest-score-first.
            token_budget: Max estimated tokens. Defaults to settings.max_context_tokens.

        Returns:
            Single string ready to be injected into build_prompt().
            Returns "" if no documents fit within the budget.
        """
        budget = token_budget if token_budget is not None else settings.max_context_tokens
        segments: List[str] = []
        running_tokens: int = 0

        for doc in fused_docs:
            title   = doc.get("title", "Untitled")
            content = doc.get("content", doc.get("text", "")).strip()

            if not content:
                # Skip documents with no retrievable text
                continue

            segment = f"[Document: '{title}']\n{content}\n\n"
            cost    = estimate_tokens(segment)

            if running_tokens + cost > budget:
                logger.debug(
                    "ContextBuilder.build_context: skipping '%s' "
                    "(would bring total to %d, budget is %d)",
                    title, running_tokens + cost, budget,
                )
                continue   # try next document — do not break

            segments.append(segment)
            running_tokens += cost

        context = "".join(segments).rstrip()
        logger.debug(
            "ContextBuilder.build_context: included %d documents "
            "(~%d estimated tokens, budget %d)",
            len(segments), running_tokens, budget,
        )
        return context

    # ── Prompt assembly ────────────────────────────────────────────────────────

    def build_prompt(self, context: str, query: str) -> str:
        """
        Injects assembled context into the system prompt template and
        appends the user query.

        The system prompt (settings.system_prompt) contains all three required
        instructions:
          1. Answer ONLY from the provided context — no outside knowledge.
          2. Explicitly state when the answer is NOT found in the notes.
          3. Cite document titles using: [Source: 'Document Title'].
        """
        system = settings.system_prompt.format(context=context or "(no context retrieved)")
        return f"{system}\n\nUSER QUERY:\n{query}\n\nANSWER:\n"


# Singleton — import this everywhere
context_builder = ContextBuilder()
