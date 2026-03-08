from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8765

    # ── Paths (all derived from base_dir — no absolute hardcodes) ─────────────
    base_dir: Path = Path(__file__).resolve().parent
    db_path: Path = base_dir / "data" / "db" / "nexus.sqlite"
    vector_db_path: Path = base_dir / "data" / "vectors"
    llm_models_path: Path = base_dir / "data" / "llm_models"
    docs_path: Path = base_dir / "data" / "documents"

    # ── Model ──────────────────────────────────────────────────────────────────
    # Filename of the GGUF model inside llm_models_path.
    # Override via DEFAULT_MODEL env-var or .env file.
    default_model: str = "Phi-3-mini-4k-instruct-q4.gguf"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Retrieval hyper-parameters ─────────────────────────────────────────────
    rrf_k: int = 60          # RRF rank-fusion constant
    top_k: int = 20          # Candidate blocks from vector search
    top_n: int = 8           # Final documents after RRF re-rank
    max_context_tokens: int = 2048   # Token budget for context window
    graph_hop_depth: int = 2         # BFS hops during graph traversal

    # ── Ingest ─────────────────────────────────────────────────────────────────
    pdf_chunk_size: int = 2000       # Characters per PDF chunk (≈ 512 tokens)
    max_file_size_mb: int = 50       # Maximum upload size in MB

    # ── Debug / Dev ────────────────────────────────────────────────────────────
    # Set DEBUG_MODE=True in .env during development only.
    # Controls SQLAlchemy echo and log verbosity.
    debug_mode: bool = False

    # ── System prompt ──────────────────────────────────────────────────────────
    system_prompt: str = (
        "You are an AI assistant for NexusAir knowledge base.\n"
        "Answer ONLY using the context provided below from the user's documents. "
        "Do NOT use any outside knowledge.\n"
        "If the answer is not found in the provided context, respond with: "
        "\"I don't have information about that in your notes.\"\n"
        "Always cite the specific document title(s) you are referencing "
        "using the format: [Source: 'Document Title'].\n"
        "Be concise. Target 200 words unless the question requires more.\n\n"
        "CONTEXT:\n{context}"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere
settings = Settings()
