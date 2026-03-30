"""Disk-based caching for expensive API calls."""

import hashlib
import json
import logging
import time
from functools import wraps

from .config import DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / ".cache"


def disk_cache(ttl_hours: float = 4):
    """Cache function results to disk with a TTL.

    Uses JSON serialization. Cache key is derived from function name + args.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key_parts = f"{fn.__module__}.{fn.__name__}:{args}:{sorted(kwargs.items())}"
            key_hash = hashlib.md5(key_parts.encode()).hexdigest()[:12]
            key = f"{fn.__name__}_{key_hash}"
            path = CACHE_DIR / f"{key}.json"

            if path.exists():
                age_hours = (time.time() - path.stat().st_mtime) / 3600
                if age_hours < ttl_hours:
                    try:
                        result = json.loads(path.read_text())
                        logger.debug("Cache hit for %s (age %.1fh)", fn.__name__, age_hours)
                        return result
                    except (json.JSONDecodeError, ValueError):
                        logger.debug("Cache read failed for %s, refetching", fn.__name__)

            result = fn(*args, **kwargs)

            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(result, default=str))
                tmp.rename(path)
            except (TypeError, OSError):
                logger.debug("Cache write failed for %s", fn.__name__)

            return result
        wrapper.uncached = fn
        return wrapper
    return decorator
