"""
catalog.py

Loads the SHL product catalog from the JSON file and builds
lookup structures we need throughout the app.

I'm keeping this separate from the search logic so it's easy
to swap the catalog source later (e.g. re-scrape, update file).
"""

import json
import os
from typing import Optional

# Map full key names to the short letter codes used in the API response
# These codes come directly from the sample conversations
KEY_CODE_MAP = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

# Where the catalog file lives — relative to this file
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")


def _load_raw(path: str) -> list[dict]:
    """
    Read the catalog JSON. The raw file from SHL has some literal
    newline characters inside string values (e.g. 'Microsoft \n 365')
    which breaks standard JSON parsing. We fix those before parsing.
    """
    with open(path, "rb") as f:
        raw = f.read()

    text = raw.decode("utf-8", errors="replace")

    # walk char by char and replace bare newlines inside strings with a space
    # it's a bit verbose but it's explicit and easy to debug
    result = []
    in_string = False
    for ch in text:
        if ch == '"' and (not result or result[-1] != "\\"):
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append(" ")
        else:
            result.append(ch)

    return json.loads("".join(result))


def _make_test_type_str(keys: list[str]) -> str:
    """
    Turn ['Knowledge & Skills', 'Simulations'] into 'K,S'
    This is the test_type field in our API responses.
    """
    codes = [KEY_CODE_MAP.get(k, "?") for k in keys]
    return ",".join(codes)


def _make_search_text(item: dict) -> str:
    """
    Build a single string we'll embed for semantic search.
    Combines name + description + keys + job_levels so that
    queries about role type, skill area, or assessment type
    all find the right results.
    """
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
        " ".join(item.get("languages", [])),
    ]
    return " ".join(p for p in parts if p).strip()


class Catalog:
    """
    Holds all 377 assessments in memory.

    After loading:
      - self.items         -> list of cleaned dicts
      - self.by_link       -> dict keyed by URL for fast lookup / validation
      - self.search_texts  -> list of strings parallel to self.items, used for embedding
    """

    def __init__(self, path: str = CATALOG_PATH):
        raw = _load_raw(path)

        self.items = []
        self.by_link: dict[str, dict] = {}
        self.search_texts: list[str] = []

        for entry in raw:
            # skip anything without a valid link (shouldn't happen but be safe)
            link = entry.get("link", "").strip()
            if not link:
                continue

            item = {
                "entity_id": entry.get("entity_id", ""),
                "name": entry.get("name", "").strip(),
                "url": link,
                "description": entry.get("description", "").strip(),
                "test_type": _make_test_type_str(entry.get("keys", [])),
                "keys": entry.get("keys", []),
                "job_levels": entry.get("job_levels", []),
                "languages": entry.get("languages", []),
                "duration": entry.get("duration", "").strip(),
                "remote": entry.get("remote", "yes"),
                "adaptive": entry.get("adaptive", "no"),
            }

            self.items.append(item)
            self.by_link[link] = item
            self.search_texts.append(_make_search_text(item))

    def is_valid_url(self, url: str) -> bool:
        """Check if a URL actually exists in our catalog."""
        return url in self.by_link

    def get_by_url(self, url: str) -> Optional[dict]:
        return self.by_link.get(url)

    def __len__(self):
        return len(self.items)


# module-level singleton so we only load once at startup
_catalog: Optional[Catalog] = None


def get_catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        _catalog = Catalog()
    return _catalog
