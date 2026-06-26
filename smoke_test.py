"""Quick check that the model layer works. Run from repo root: python smoke_test.py"""
from momento.config import get_backend, BACKEND

backend = get_backend()
print(f"Backend: {BACKEND}\n")

reply = backend.chat([{"role": "user",
                       "content": "In one sentence, what is Kyoto known for?"}])
print("chat() ->", reply.strip(), "\n")

data = backend.complete_json([{"role": "user", "content":
    'Return ONLY JSON: {"city": "Kyoto", "country": "?"} with country filled in.'}])
print("complete_json() ->", data, "\n")

vecs = backend.embed(["temple", "ramen shop"])
print(f"embed() -> {len(vecs)} vectors, dim={len(vecs[0])}")