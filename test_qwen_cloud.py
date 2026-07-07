"""Prove momento's model layer works against Qwen Cloud: chat, JSON
classification, and 768-dim embeddings. Costs a few hundred tokens total.

Run from repo root:  MODEL_BACKEND=qwen_cloud python test_qwen_cloud.py
"""
import os
os.environ["MODEL_BACKEND"] = "qwen_cloud"   # force cloud for this process

from momento.config import get_backend, QWEN_CLOUD

assert QWEN_CLOUD["api_key"], "QWEN_API_KEY missing from .env"
backend = get_backend()
print(f"Backend: qwen_cloud  ({QWEN_CLOUD['base_url']})")

print("\n[1/3] chat (qwen3.7-plus)...")
reply = backend.chat([{"role": "user",
                       "content": "In one sentence, what is Kyoto known for?"}],
                     max_tokens=100)
print("   ->", reply.strip()[:120])

print("\n[2/3] complete_json (qwen-flash)...")
data = backend.complete_json([{"role": "user", "content":
    'Return ONLY JSON: {"city": "Kyoto", "country": "?"} with country filled in.'}],
    fast=True)
print("   ->", data)
assert data.get("country"), "JSON parse failed"

print("\n[3/3] embed (text-embedding-v4, dimensions=768)...")
vecs = backend.embed(["temple", "ramen shop"])
print(f"   -> {len(vecs)} vectors, dim={len(vecs[0])}")
assert len(vecs[0]) == 768, f"expected 768 dims, got {len(vecs[0])} — dimension mismatch!"

print("\nALL THREE CLOUD CALLS OK — prod parity confirmed.")