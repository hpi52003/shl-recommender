import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from catalog import get_catalog

# Using MiniLM — good balance of speed and quality for this task
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_index: faiss.IndexFlatIP | None = None  # Inner product (cosine on normalized vecs)
_built = False


def _build_index():
    """
    Load the embedding model and build the FAISS index.
    Called once at startup. Takes ~5-10 seconds.
    """
    global _model, _index, _built

    catalog = get_catalog()
    _model = SentenceTransformer(MODEL_NAME)

    texts = catalog.search_texts
    # encode all catalog entries
    embeddings = _model.encode(texts, show_progress_bar=False, batch_size=64)

    # normalize so inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-10, None)

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(embeddings.astype(np.float32))

    _built = True
    print(f"[search] FAISS index built: {len(texts)} items, dim={dim}")


def ensure_index():
    """Call this at startup to warm up the index before the first request."""
    if not _built:
        _build_index()


def retrieve(query: str, top_k: int = 20) -> list[dict]:
    """
    Given a natural language query, return up to top_k catalog items
    ranked by semantic similarity.

    Returns a list of catalog item dicts (same structure as catalog.items).
    """
    if not _built:
        _build_index()

    catalog = get_catalog()

    # embed the query
    q_vec = _model.encode([query], show_progress_bar=False)
    q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
    q_vec = (q_vec / np.clip(q_norm, 1e-10, None)).astype(np.float32)

    # search
    scores, indices = _index.search(q_vec, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        item = catalog.items[idx].copy()
        item["_score"] = float(score)
        results.append(item)

    return results


def retrieve_filtered(
    query: str,
    top_k: int = 20,
    job_levels: list[str] | None = None,
    key_types: list[str] | None = None,
) -> list[dict]:
    """
    Same as retrieve() but with optional post-filters.

    job_levels: e.g. ['Graduate', 'Entry-Level'] — returns items that
                match ANY of the given levels
    key_types:  e.g. ['Personality & Behavior'] — returns items that
                include ANY of the given types

    We fetch more candidates (top_k * 3) before filtering so we
    don't end up with an empty result set after filtering.
    """
    # fetch more to account for filtering drop-off
    candidates = retrieve(query, top_k=top_k * 3)

    filtered = []
    for item in candidates:
        if job_levels:
            item_levels = set(item.get("job_levels", []))
            if not item_levels.intersection(job_levels):
                continue
        if key_types:
            item_keys = set(item.get("keys", []))
            if not item_keys.intersection(key_types):
                continue
        filtered.append(item)
        if len(filtered) >= top_k:
            break

    # if filtering gave us nothing, fall back to unfiltered
    # better to return something than nothing
    if not filtered:
        filtered = candidates[:top_k]

    return filtered
