"""Answer cache for completed (non-RFI) answers.

Keyed on the normalised question plus the settings that change the answer (model, effort, search
mode, top_k), so a config change never serves a stale answer. Redis when REDIS_URL is set, so
several API replicas share one cache; otherwise an in-process dict with the same TTL semantics.
"""

import hashlib
import json
import re
import time

from sitebrain.config import Settings


def cache_key(question: str, settings: Settings) -> str:
    normalised = re.sub(r"\s+", " ", question.strip().lower())
    parts = [normalised, settings.provider, settings.model, settings.effort, settings.search_mode, str(settings.top_k), settings.sources]
    return "sitebrain:answer:" + hashlib.sha256("|".join(parts).encode()).hexdigest()


class MemoryCache:
    def __init__(self):
        self._data: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> dict | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, payload = item
        if expires < time.time():
            del self._data[key]
            return None
        return json.loads(payload)

    def set(self, key: str, value: dict, ttl: int) -> None:
        self._data[key] = (time.time() + ttl, json.dumps(value))


class RedisCache:
    def __init__(self, url: str):
        import redis

        self._r = redis.Redis.from_url(url)

    def get(self, key: str) -> dict | None:
        raw = self._r.get(key)
        return json.loads(raw) if raw else None

    def set(self, key: str, value: dict, ttl: int) -> None:
        self._r.set(key, json.dumps(value), ex=ttl)


def make_cache(settings: Settings) -> MemoryCache | RedisCache | None:
    if settings.cache_ttl <= 0:
        return None
    return RedisCache(settings.redis_url) if settings.redis_url else MemoryCache()
