"""Exercise Loop 1: predictive prefetch, current-vs-predicted pulls, subject
scoping, the token budget, cross-turn prediction, miss->fallback, and the
Loop1->Loop2 promotion link. From repo root: python test_prefetch.py"""
from datetime import datetime, timezone, timedelta

from momento.config import get_backend
from momento.memory.store import MemoryStore
from momento.memory.schema import MemoryRecord, FactType, Provenance, Tier
from momento.memory.prefetch import Prefetcher
from momento.intent.classifier import classify
from momento.intent.transitions import top_intents
from momento.intent.taxonomy import Intent

backend = get_backend()
store = MemoryStore(backend, db_path=":memory:")
pf = Prefetcher(store)


def days_ago(n):
    return datetime.now(timezone.utc) - timedelta(days=n)


def seed(**kw):
    return store.add(MemoryRecord(provenance=Provenance(source="seed"), **kw))


# user-global prefs (will be HOT)
seed(text="User is vegetarian and avoids crowded tourist spots.",
     fact_type=FactType.PREFERENCE, intents=["dining", "attractions"], confidence=0.95)
seed(text="User prefers small boutique stays over large hotel chains.",
     fact_type=FactType.USER_PROFILE, intents=["accommodation"], confidence=0.9)
# Kyoto facts
seed(text="Nishiki Market is a covered food street in central Kyoto.",
     fact_type=FactType.POI_FACT, subject="Kyoto", intents=["dining", "attractions"], confidence=0.6)
seed(text="Gion is a historic Kyoto district known for geisha culture.",
     fact_type=FactType.POI_FACT, subject="Kyoto", intents=["attractions", "accommodation"], confidence=0.6)
# Japan entry — one fresh, one long-stale (must never surface)
seed(text="US citizens may enter Japan visa-free for up to 90 days.",
     fact_type=FactType.VISA_RULE, subject="Japan", intents=["entry_requirements"],
     confidence=0.85, observed_at=days_ago(3))
seed(text="Japan requires a pre-arrival quarantine declaration form.",
     fact_type=FactType.VISA_RULE, subject="Japan", intents=["entry_requirements"],
     confidence=0.85, observed_at=days_ago(220))
# a DIFFERENT subject — must not leak into a Kyoto turn
seed(text="The Louvre is a major art museum in Paris.",
     fact_type=FactType.POI_FACT, subject="Paris", intents=["attractions"], confidence=0.6)

print("=" * 70)
print("TURN: 'Where should we grab dinner in Kyoto?'  (intent should be dining)")
turn = "Where should we grab dinner in Kyoto?"
c = classify(turn, backend)
res = pf.prefetch(c, query=turn)          # subject auto-detected from entities
print(res.summary())
print("\n" + res.text())

texts = " ".join(r.text for r in res.buffer)
print("\nchecks:")
print("  vegetarian pref warmed (current intent + global):", "vegetarian" in texts)
print("  Nishiki warmed (dining · Kyoto):", "Nishiki" in texts)
print("  Gion warmed via PREDICTED intent (attractions):", "Gion" in texts)
print("  Paris fact EXCLUDED (wrong subject):", "Louvre" not in texts)
print("  stale visa EXCLUDED (retired):", "quarantine" not in texts)
print("  under 400-token budget:", res.token_estimate <= 400)

print("\n" + "=" * 70)
print("CROSS-TURN PREDICTION: last turn framed the trip; this turn -> accommodation")
predicted_after_framing = top_intents(Intent.TRIP_FRAMING, k=3)
print("  predicted after trip_framing:", [i.value for i in predicted_after_framing])
turn2 = "What area should we stay in?"
c2 = classify(turn2, backend)
res2 = pf.prefetch(c2, query=turn2, subject="Kyoto",
                   predicted_last_turn=predicted_after_framing)
print(f"  this turn intent={res2.current_intent.value}  "
      f"prediction_hit={res2.prediction_hit}  (was it pre-warmed last turn?)")

print("\n" + "=" * 70)
print("MISS + FALLBACK: ask about transport (no transport memories stored)")
c3 = classify("How do the trains work for getting around?", backend)
res3 = pf.prefetch(c3, query="How do the trains work for getting around?", subject="Kyoto")
print(f"  intent={res3.current_intent.value}  intent_hit={res3.intent_hit}  "
      f"needs_fallback={res3.needs_fallback}")
fb = pf.fallback_search("getting around Kyoto", k=2)
print("  reactive fallback still returns nearest memories:",
      [r.text[:30] for r, _ in fb])

print("\n" + "=" * 70)
print("LOOP 1 -> LOOP 2: promote-on-use raises score/tier of warmed Kyoto facts")
nishiki_id = next(r.id for r in store.all() if "Nishiki" in r.text)
before = store.get(nishiki_id)
print(f"  Nishiki before: hits={before.hit_count} score={before.score:.3f} tier={before.tier.value}")
for _ in range(5):
    pf.prefetch(classify("good dinner spots in Kyoto", backend),
                query="good dinner spots in Kyoto", subject="Kyoto", promote=True)
after = store.get(nishiki_id)
print(f"  Nishiki after 5: hits={after.hit_count} score={after.score:.3f} tier={after.tier.value}")
print("  promoted toward HOT:", after.hit_count >= 5 and after.score > before.score)

print("\n" + "=" * 70)
print("TOKEN BUDGET PRESSURE: same turn, tiny 60-token buffer -> trims to top priority")
tight = Prefetcher(store, token_budget=60)
rt = tight.prefetch(classify(turn, backend), query=turn)
print(f"  fit {len(rt.buffer)} memories in ~{rt.token_estimate} tok "
      f"(hot={rt.counts['hot']} current={rt.counts['current']} predicted={rt.counts['predicted']})")

store.close()