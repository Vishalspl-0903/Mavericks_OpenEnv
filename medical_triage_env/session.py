"""
Session management for medical triage environment.

Provides thread-safe session lifecycle management with TTL-based eviction.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Dict, List
from uuid import uuid4

import structlog
from fastapi import HTTPException

# Import will be updated once env.py is refactored to accept task_id
# For now, we'll use TYPE_CHECKING to avoid circular imports during development
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .env import MedicalTriageEnv

logger = structlog.get_logger(__name__)


class SessionManager:
    """Singleton session manager with TTL-based eviction and thread-safe operations."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize session store and TTL tracking."""
        if self._initialized:
            return

        self._sessions: Dict[str, MedicalTriageEnv] = {}
        self._last_access: Dict[str, float] = {}
        self._ttl_seconds = 1800  # 30 minutes
        self._sweep_task: asyncio.Task | None = None
        self._initialized = True

        logger.info("session_manager_initialized", ttl_seconds=self._ttl_seconds)

    def create(self, task_id: str) -> str:
        """Create a new session and return its ID."""
        # Lazy import to avoid circular dependency
        from .env import MedicalTriageEnv

        session_id = str(uuid4())

        with self._lock:
            # Create new env instance - will be updated in Step 6 to accept task_id
            env = MedicalTriageEnv()
            # Initialize with task_id via reset for now
            env.reset(task_id)
            self._sessions[session_id] = env
            self._last_access[session_id] = time.time()

        logger.info(
            "session_created",
            session_id=session_id,
            task_id=task_id,
            active_sessions=len(self._sessions),
        )

        return session_id

    def get(self, session_id: str) -> MedicalTriageEnv:
        """Retrieve a session by ID, raising 404 if not found."""
        with self._lock:
            if session_id not in self._sessions:
                logger.warning("session_not_found", session_id=session_id)
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_id} not found or expired",
                )

            # Update last access time
            self._last_access[session_id] = time.time()
            return self._sessions[session_id]

    def destroy(self, session_id: str) -> None:
        """Remove a session from the store."""
        with self._lock:
            if session_id in self._sessions:
                env = self._sessions[session_id]
                task_id = env.current_task["task_id"] if env.current_task else "unknown"
                del self._sessions[session_id]
                del self._last_access[session_id]

                logger.info(
                    "session_destroyed",
                    session_id=session_id,
                    task_id=task_id,
                    active_sessions=len(self._sessions),
                )

    def list_active(self) -> List[str]:
        """Return list of active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def _evict_expired_sessions(self) -> None:
        """Internal method to evict sessions that exceeded TTL."""
        current_time = time.time()
        expired_sessions = []

        with self._lock:
            for session_id, last_access in self._last_access.items():
                if current_time - last_access > self._ttl_seconds:
                    expired_sessions.append(session_id)

        # Destroy outside the iteration to avoid dict size change issues
        for session_id in expired_sessions:
            logger.info(
                "session_expired",
                session_id=session_id,
                ttl_seconds=self._ttl_seconds,
            )
            self.destroy(session_id)

        if expired_sessions:
            logger.info("ttl_sweep_completed", evicted_count=len(expired_sessions))

    async def _ttl_sweep_loop(self) -> None:
        """Background task that runs TTL sweep every 5 minutes."""
        sweep_interval = 300  # 5 minutes

        logger.info("ttl_sweep_loop_started", interval_seconds=sweep_interval)

        while True:
            try:
                await asyncio.sleep(sweep_interval)
                self._evict_expired_sessions()
            except asyncio.CancelledError:
                logger.info("ttl_sweep_loop_cancelled")
                break
            except Exception as e:
                logger.error(
                    "ttl_sweep_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )

    def start_ttl_sweep(self) -> None:
        """Start the background TTL sweep task (asyncio-compatible)."""
        if self._sweep_task is None or self._sweep_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._sweep_task = loop.create_task(self._ttl_sweep_loop())
                logger.info("ttl_sweep_started")
            except RuntimeError:
                logger.warning("ttl_sweep_not_started_no_event_loop")

    def stop_ttl_sweep(self) -> None:
        """Stop the background TTL sweep task."""
        if self._sweep_task and not self._sweep_task.done():
            self._sweep_task.cancel()
            logger.info("ttl_sweep_stopped")
