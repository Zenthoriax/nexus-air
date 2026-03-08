"""
InferenceService — llama-cpp-python GGUF model wrapper.

Key fixes vs original:
  - threading.Event for clean stream cancellation checked per-token
  - SHA-256 sidecar verification before loading
  - n_threads=os.cpu_count() for maximum CPU utilisation
  - Structured logging replacing print()
  - Clear FileNotFoundError message on missing model
  - stream_response() resets stop event at call start
"""

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Generator, Any

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self):
        self._model = None
        self._stop_event = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load_model(self, model_path: str) -> None:
        """
        Loads the GGUF model from disk.
        Raises FileNotFoundError with a helpful message if absent.
        Optionally verifies SHA-256 if a sidecar .sha256 file exists.
        Called from lifespan via run_in_executor — stays synchronous.
        """
        from llama_cpp import Llama  # imported here so the class is importable without llama_cpp installed

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Download it via the app Settings or run download_model.py."
            )

        # Optional SHA-256 verification
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.exists():
            expected = sidecar.read_text().strip().lower()
            logger.info("InferenceService: verifying model integrity against %s", sidecar.name)
            actual = self._sha256(path)
            if actual != expected:
                raise ValueError(
                    f"Model file failed SHA-256 check.\n"
                    f"  Expected: {expected}\n"
                    f"  Actual:   {actual}\n"
                    f"  Path:     {model_path}"
                )
            logger.info("InferenceService: SHA-256 verified OK")
        else:
            digest = self._sha256(path)
            logger.info("InferenceService: model SHA-256 (no sidecar): %s", digest)

        logger.info("InferenceService: loading model from %s", model_path)
        self._model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=0,                    # CPU-only
            n_threads=os.cpu_count() or 4,    # use all available cores
            verbose=False,                     # suppress llama.cpp C-level stdout
        )
        logger.info("InferenceService: model loaded OK")

    # ── Inference ──────────────────────────────────────────────────────────────

    def stream_response(self, prompt: str, max_tokens: int = 1024) -> Generator[Any, None, None]:
        """
        Returns a synchronous token generator from llama-cpp-python.

        IMPORTANT: This is a blocking C-level generator.  The caller MUST
        run it inside a ThreadPoolExecutor to avoid blocking the asyncio loop.

        The stop event is cleared at the start of each call so that a
        previous /stop flag does not bleed into the next request.
        """
        if not self.is_loaded:
            from exceptions import ModelNotLoadedError
            raise ModelNotLoadedError()

        self._stop_event.clear()

        for chunk in self._model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
        ):
            if self._stop_event.is_set():
                logger.info("InferenceService: stream cancelled via stop event")
                return
            yield chunk

    def stop(self) -> None:
        """Sets the cancellation flag. The generator checks it each token."""
        logger.info("InferenceService: stop requested")
        self._stop_event.set()

    # ── Status ─────────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


# Singleton
inference_service = InferenceService()
