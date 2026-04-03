"""Cache embeddings for the 10 Pegasus descriptions. Uses 10 API calls."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from __init__ import get_client, embed_text, _EMBED_CACHE, _DESC_CACHE, _load_cache

desc_cache = _load_cache(_DESC_CACHE)
emb_cache = _load_cache(_EMBED_CACHE)

client = get_client()
total = len(desc_cache)

for i, (fhash, desc) in enumerate(desc_cache.items()):
    key = desc[:500]
    if key in emb_cache:
        print(f"  [{i+1}/{total}] Already cached, skipping")
        continue
    if desc.startswith("Error:"):
        print(f"  [{i+1}/{total}] Skipping error description")
        continue
    print(f"  [{i+1}/{total}] Embedding: {desc[:60]}...")
    embed_text(client, desc)  # this saves to cache automatically

print(f"\nDone! Cached embeddings: {len(_load_cache(_EMBED_CACHE))}")
