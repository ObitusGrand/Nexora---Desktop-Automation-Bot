"""Encrypted local workflow memory backed by ChromaDB."""

from __future__ import annotations

import base64
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class WorkflowTrace:
    """Replayable actions and non-sensitive execution metadata."""

    task_prompt: str
    actions: list[dict[str, Any]]
    app_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowMatch:
    trace: WorkflowTrace
    similarity: float


class WorkflowMemory(Protocol):
    def find(self, prompt: str, threshold: float) -> WorkflowMatch | None: ...
    def save(self, trace: WorkflowTrace) -> None: ...


class MemorySecurityError(RuntimeError):
    """Raised when encrypted workflow memory cannot be initialized safely."""


class ChromaWorkflowMemory:
    """Persist workflow embeddings in Chroma and encrypt all stored trace data.

    The prompt is embedded locally and passed as an embedding only. Chroma never
    receives it as a document, while the replay payload is encrypted with Fernet.
    """

    def __init__(
        self,
        path: str | Path,
        key: bytes | None = None,
        collection_name: str = "desktop_bot_workflows",
    ) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise MemorySecurityError(
                "Phase 4 requires chromadb and cryptography"
            ) from exc

        self._fernet_type = Fernet
        self._fernet = Fernet(key or self._load_or_create_key(Path(path) / "memory.key"))
        embedding = DefaultEmbeddingFunction()
        self._embedding = embedding
        self._collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
            name=collection_name,
            embedding_function=embedding,
        )

    @staticmethod
    def _load_or_create_key(path: Path) -> bytes:
        configured = os.environ.get("DESKTOP_BOT_MEMORY_KEY")
        if configured:
            return configured.encode("ascii")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return path.read_bytes()
        try:
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
        except ImportError as exc:
            raise MemorySecurityError("cryptography is required for workflow encryption") from exc
        path.write_bytes(key)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return key

    def find(self, prompt: str, threshold: float) -> WorkflowMatch | None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        embedding = self._embedding([prompt])[0]
        result = self._collection.query(query_embeddings=[embedding], n_results=1)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        if not ids or not distances or not metadatas:
            return None
        raw_distance = float(distances[0])
        # Chroma default embedding uses L2; clamp to [0, 1] for threshold comparison
        similarity = max(0.0, min(1.0, 1.0 - raw_distance / 2.0))
        if similarity < threshold:
            return None
        payload = self._fernet.decrypt(str(metadatas[0]["payload"]).encode("ascii"))
        return WorkflowMatch(WorkflowTrace(**json.loads(payload)), similarity)

    def save(self, trace: WorkflowTrace) -> None:
        payload = self._fernet.encrypt(json.dumps(asdict(trace)).encode("utf-8")).decode("ascii")
        embedding = self._embedding([trace.task_prompt])[0]
        self._collection.add(
            ids=[uuid.uuid4().hex],
            embeddings=[embedding],
            metadatas=[{"payload": payload}],
        )