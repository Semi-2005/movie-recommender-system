"""
TTL-Based In-Memory Cache
=========================

Provides a lightweight, thread-safe TTL cache decorator for
recommendation model results.

Uses ``cachetools.TTLCache`` when available, otherwise falls back
to a custom dict + timestamp implementation so ``cachetools`` is
not a hard dependency.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from cachetools import TTLCache

    _HAS_CACHETOOLS = True
    logger.info("cachetools available — using TTLCache backend")
except ImportError:
    _HAS_CACHETOOLS = False
    logger.info("cachetools not installed — using built-in TTL cache")


def _make_key(*args: Any, **kwargs: Any) -> str:
    """
    Build a deterministic cache key from function arguments.

    Uses a SHA-256 hash of the string representation to keep keys
    fixed-length and hashable regardless of argument types.
    """
    raw = f"{args}|{sorted(kwargs.items())}"
    return hashlib.sha256(raw.encode()).hexdigest()


class _BuiltinTTLCache:
    """
    Minimal TTL cache using a plain dict + timestamps.

    Thread-safe via a reentrant lock. Performs lazy eviction on
    ``get`` — expired entries are removed when accessed, not on a
    background timer (simple and predictable).
    """

    def __init__(self, maxsize: int, ttl: float) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            timestamp, value = entry
            if time.monotonic() - timestamp > self._ttl:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._data) >= self._maxsize and key not in self._data:
                self._evict_expired()
                if len(self._data) >= self._maxsize:
                    # Remove the oldest entry
                    oldest_key = min(
                        self._data, key=lambda k: self._data[k][0]
                    )
                    del self._data[oldest_key]

            self._data[key] = (time.monotonic(), value)

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        now = time.monotonic()
        expired = [
            k for k, (ts, _) in self._data.items()
            if now - ts > self._ttl
        ]
        for k in expired:
            del self._data[k]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._data)


def ttl_cache(
    maxsize: int = 5000,
    ttl_seconds: float = 1800,
) -> Callable:
    """
    Decorator that caches function return values with TTL eviction.

    Args:
        maxsize:     Maximum number of cached entries.
        ttl_seconds: Time-to-live in seconds (default: 30 minutes).

    Returns:
        A decorator that wraps the target function with caching.

    Example::

        @ttl_cache(maxsize=1000, ttl_seconds=900)
        def expensive_computation(movie_id: int) -> dict:
            ...
    """
    if _HAS_CACHETOOLS:
        cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        lock = threading.RLock()
    else:
        cache = _BuiltinTTLCache(maxsize=maxsize, ttl=ttl_seconds)
        lock = None  # _BuiltinTTLCache is internally thread-safe

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _make_key(*args, **kwargs)

            if _HAS_CACHETOOLS:
                with lock:
                    cached = cache.get(key)
                    if cached is not None:
                        return cached
            else:
                cached = cache.get(key)
                if cached is not None:
                    return cached

            result = func(*args, **kwargs)

            if _HAS_CACHETOOLS:
                with lock:
                    cache[key] = result
            else:
                cache.set(key, result)

            return result

        # Expose cache control for testing / hot-reload
        wrapper.cache_clear = (
            cache.clear if hasattr(cache, "clear")
            else lambda: None
        )
        wrapper.cache_info = lambda: {
            "backend": "cachetools.TTLCache" if _HAS_CACHETOOLS else "builtin",
            "maxsize": maxsize,
            "ttl_seconds": ttl_seconds,
            "current_size": len(cache),
        }

        return wrapper
    return decorator
