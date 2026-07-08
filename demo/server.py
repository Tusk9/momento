"""momento demo server — wraps the real Agent behind a tiny JSON API.
  local:  python demo/server.py
  cloud:  MODEL_BACKEND=qwen_cloud python demo/server.py     (the judged run)
Then open http://localhost:5001
"""
import os, sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from momento.config import get_backend, BACKEND
from momento.memory.store import MemoryStore
from momento.memory.schema import MemoryRecord, FactType, Provenance
from momento.agent import Agent

DB = "data/demo.db"
assert os.path.exists(DB), "Run `python demo/seed_demo.py` first."

backend = get_backend()
store = MemoryStore(backend, db_path=DB)
agent = Agent(store, backend)

app = Flask(__name__, static_folder="static")


def mem_json(r: MemoryRecord) -> dict:
    return {
        "id": r.id, "text": r.text, "fact_type": r.fact_type.value,
        "kind": r.kind.value, "tier": r.tier.value, "score": round(r.score, 3),
        "hit_count": r.hit_count, "subject": r.subject,
        "lat": r.lat, "lon": r.lon, "active": r.is_active,
        "superseded_by": r.superseded_by,
        "source": r.provenance.source if r.provenance else None,
        "source_url": r.provenance.detail if r.provenance else None,
    }


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/memories")
def memories():
    recs = store.all(include_inactive=True)
    return jsonify({"backend": BACKEND, "memories": [mem_json(r) for r in recs]})


@app.post("/api/turn")
def turn():
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "empty message"}), 400
    t = agent.turn(text)
    return jsonify({
        "reply": t.reply,
        "intent": t.classification.intent.value,
        "subject": t.subject,
        "prefetch": {
            "buffer": len(t.prefetch.buffer),
            "tokens": t.prefetch.token_estimate,
            "intent_hit": t.prefetch.intent_hit,
            "prediction_hit": t.prefetch.prediction_hit,
            "predicted": [i.value for i in t.prefetch.predicted_intents],
            "fallback": t.used_fallback,
        },
        "promoted": [r.text[:60] for r in t.promoted],
        "learned": [r.text for r in t.learned],
        "events": [e.caption() for e in t.ingest_events],
    })


@app.post("/api/new_session")
def new_session():
    agent.new_session()
    return jsonify({"ok": True})


@app.post("/api/inject_visa_change")
def inject():
    """Plays the data pipeline's role: a freshly re-scraped advisory arrives."""
    fresh = MemoryRecord(
        text="US citizens now require an approved eVisa before entering Japan; "
             "visa-free entry has ended.",
        fact_type=FactType.VISA_RULE, subject="Japan",
        intents=["entry_requirements"], confidence=0.85,
        observed_at=datetime.now(timezone.utc),
        provenance=Provenance(source="gov_advisory",
                              detail="https://travel.state.gov/..."))
    ev = agent.reconciler.reconcile(fresh)
    return jsonify({"outcome": ev.outcome.value, "caption": ev.caption()})


if __name__ == "__main__":
    print(f"momento demo on http://localhost:5001  (backend: {BACKEND})")
    app.run(port=5001, debug=False)