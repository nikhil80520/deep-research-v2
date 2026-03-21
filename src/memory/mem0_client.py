import os
import logging

logger = logging.getLogger(__name__)

_client = None
_enabled = False


def init_memory() -> bool:
    global _client, _enabled
    api_key = os.getenv("MEM0_API_KEY", "")
    if not api_key:
        logger.warning("MEM0_API_KEY not set — memory disabled.")
        return False
    try:
        from mem0 import MemoryClient
        _client = MemoryClient(api_key=api_key)
        _enabled = True
        return True
    except Exception as e:
        logger.error(f"mem0 init failed: {e}")
        return False


def add_memory(user_id: str, query: str, verdict: str):
    if not _enabled or not _client:
        return
    try:
        _client.add([
            {"role": "user", "content": f"Research query: {query}"},
            {"role": "assistant", "content": f"Summary: {verdict[:300]}"},
        ], user_id=user_id)
    except Exception as e:
        logger.error(f"mem0 add failed: {e}")


def search_memory(user_id: str, query: str, limit: int = 3) -> str:
    if not _enabled or not _client:
        return ""
    try:
        results = _client.search(query, user_id=user_id, limit=limit)
        memories = results.get("results", [])
        if not memories:
            return ""
        return "## Relevant Past Research\n" + "\n".join(f"- {m['memory']}" for m in memories)
    except Exception as e:
        logger.error(f"mem0 search failed: {e}")
        return ""


def get_all_memories(user_id: str) -> list:
    if not _enabled or not _client:
        return []
    try:
        result = _client.get_all(user_id=user_id)
        return result.get("results", [])
    except Exception as e:
        logger.error(f"mem0 get_all failed: {e}")
        return []
