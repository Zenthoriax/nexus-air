"""
Custom exceptions for NexusAir backend.

All domain errors subclass NexusAirError so routers can catch the base
class as a catch-all while still mapping specific subtypes to precise
HTTP status codes.
"""


class NexusAirError(Exception):
    """Base class for all NexusAir domain errors."""


class DocumentNotFoundError(NexusAirError):
    """Raised when a document ID does not exist in the database."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        super().__init__(f"Document not found: {doc_id}")


class DuplicateDocumentError(NexusAirError):
    """Raised when content_hash already exists — prevents double-ingest."""

    def __init__(self, existing_id: str):
        self.existing_id = existing_id
        super().__init__(f"Duplicate document; existing id: {existing_id}")


class ModelNotLoadedError(NexusAirError):
    """Raised when a request requires the LLM but it has not been loaded yet."""

    def __init__(self, message: str = "LLM model is not loaded yet. Try again in a moment."):
        super().__init__(message)


class ModelIntegrityError(NexusAirError):
    """Raised when a model file fails SHA-256 verification."""

    def __init__(self, path: str):
        super().__init__(f"Model file failed integrity check: {path}")


class RetrievalError(NexusAirError):
    """Raised when the retrieval pipeline fails or times out."""


class IngestError(NexusAirError):
    """Raised for unrecoverable errors during document ingestion."""


class VectorServiceUnavailableError(NexusAirError):
    """Raised when LanceDB is unavailable and cannot serve search requests."""
