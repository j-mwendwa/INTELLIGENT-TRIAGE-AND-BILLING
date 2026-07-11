"""
src/memory/entity_memory.py — Per-thread durable key-value store.

Each conversation thread gets its own JSON file under data/memory/{thread_id}.json.
When ``settings.memory_encryption_key`` is set the file contents are encrypted with
Fernet symmetric encryption and prefixed with the sentinel ``ENCRYPTED_V1:``.

Usage::

    from src.memory.entity_memory import EntityMemory

    mem = EntityMemory(thread_id="thread-abc123")
    mem.remember("patient_name", "Alice Wanjiru")
    mem.update({"dob": "1985-03-14", "insurance": "NHIF"})
    print(mem.all())   # {"patient_name": "Alice Wanjiru", "dob": "1985-03-14", ...}
    mem.clear()        # wipes all keys but keeps the file
    mem.delete()       # removes the file entirely
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import structlog

from src.config import MEMORY_DIR, settings

logger = structlog.get_logger(__name__)

_ENCRYPTION_SENTINEL = "ENCRYPTED_V1:"


def _get_fernet():
    """Return a Fernet instance or None if encryption is not configured.

    Import is deferred so that the ``cryptography`` package is only required
    when encryption is actually used, keeping the base image lighter.
    """
    key: str = settings.memory_encryption_key
    if not key:
        return None

    try:
        from cryptography.fernet import Fernet  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "cryptography package is required for memory encryption. "
            "Install it with: pip install cryptography"
        ) from exc

    # Accept both raw base-64 Fernet keys and URL-safe base-64 strings.
    key_bytes = key.encode() if isinstance(key, str) else key
    return Fernet(key_bytes)


class EntityMemory:
    """Per-thread durable key-value store backed by a JSON file.

    Parameters
    ----------
    thread_id:
        Unique identifier for the conversation thread.  Used verbatim as the
        stem of the backing file name so it should be a safe filesystem token
        (e.g. a UUID or slugified string).
    """

    def __init__(self, thread_id: str) -> None:
        if not thread_id or not isinstance(thread_id, str):
            raise ValueError("thread_id must be a non-empty string.")

        self.thread_id: str = thread_id
        self._path: Path = MEMORY_DIR / f"{thread_id}.json"
        self._lock: threading.Lock = threading.Lock()
        self._fernet = _get_fernet()

        # Ensure the parent directory exists (config.py already does this at
        # import time, but we guard here in case of isolated test usage).
        self._path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(
            "entity_memory.init",
            thread_id=thread_id,
            path=str(self._path),
            encrypted=self._fernet is not None,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def remember(self, key: str, value: str) -> None:
        """Persist a single key-value pair.

        Parameters
        ----------
        key:
            The entity key (e.g. ``"patient_name"``).
        value:
            A string value.  Store structured data as a JSON string if needed.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string.")

        with self._lock:
            data = self._load()
            data[key] = value
            self._save(data)

        logger.info("entity_memory.remember", thread_id=self.thread_id, key=key)

    def update(self, facts: dict[str, Any]) -> None:
        """Merge *facts* into the stored entities, overwriting existing keys.

        Parameters
        ----------
        facts:
            A mapping of entity keys to string values.
        """
        if not isinstance(facts, dict):
            raise TypeError("facts must be a dict.")

        with self._lock:
            data = self._load()
            data.update(facts)
            self._save(data)

        logger.info(
            "entity_memory.update",
            thread_id=self.thread_id,
            keys=list(facts.keys()),
        )

    def all(self) -> dict[str, Any]:
        """Return a snapshot of all stored entities.

        Returns
        -------
        dict
            A shallow copy of the current entity store.
        """
        with self._lock:
            data = self._load()

        logger.debug(
            "entity_memory.all",
            thread_id=self.thread_id,
            count=len(data),
        )
        return dict(data)

    def clear(self) -> None:
        """Delete all entity keys, but keep the backing file on disk."""
        with self._lock:
            self._save({})

        logger.info("entity_memory.clear", thread_id=self.thread_id)

    def delete(self) -> None:
        """Remove the backing file entirely.

        Safe to call even if the file does not exist.
        """
        with self._lock:
            if self._path.exists():
                self._path.unlink()
                logger.info(
                    "entity_memory.delete",
                    thread_id=self.thread_id,
                    path=str(self._path),
                )
            else:
                logger.debug(
                    "entity_memory.delete.noop",
                    thread_id=self.thread_id,
                    reason="file_not_found",
                )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        """Read and decode the backing file.

        Returns an empty dict when the file does not yet exist.  Must be
        called while the caller already holds ``self._lock``.
        """
        if not self._path.exists():
            return {}

        try:
            raw: str = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error(
                "entity_memory.load.read_error",
                thread_id=self.thread_id,
                error=str(exc),
            )
            return {}

        # Decrypt if needed
        if raw.startswith(_ENCRYPTION_SENTINEL):
            if self._fernet is None:
                logger.error(
                    "entity_memory.load.decrypt_error",
                    thread_id=self.thread_id,
                    reason="file_is_encrypted_but_no_key_configured",
                )
                return {}
            try:
                ciphertext = raw[len(_ENCRYPTION_SENTINEL) :]
                raw = self._fernet.decrypt(ciphertext.encode()).decode("utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "entity_memory.load.decrypt_error",
                    thread_id=self.thread_id,
                    error=str(exc),
                )
                return {}

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(
                "entity_memory.load.json_error",
                thread_id=self.thread_id,
                error=str(exc),
            )
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        """Serialise *data* and write it to the backing file.

        Handles optional encryption.  Must be called while the caller already
        holds ``self._lock``.
        """
        try:
            payload: str = json.dumps(data, ensure_ascii=False, indent=2)

            if self._fernet is not None:
                ciphertext: str = self._fernet.encrypt(payload.encode()).decode()
                payload = f"{_ENCRYPTION_SENTINEL}{ciphertext}"

            # Atomic write: write to a temp file then rename to avoid
            # corruption if the process is killed mid-write.
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(self._path)

        except OSError as exc:
            logger.error(
                "entity_memory.save.write_error",
                thread_id=self.thread_id,
                error=str(exc),
            )
            raise

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"EntityMemory(thread_id={self.thread_id!r}, "
            f"path={str(self._path)!r}, "
            f"encrypted={self._fernet is not None})"
        )

    def __contains__(self, key: str) -> bool:
        return key in self.all()

    def __len__(self) -> int:
        return len(self.all())
