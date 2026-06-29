"""Exercise the store: add, search, LOOP-2 promotion, and passive retirement
of a stale volatile fact. From repo root: python test_store.py"""
from datetime import datetime, timezone, timedelta

from momento.config import get_backend
from momento.memory.store import MemoryStore
from momento.memory.schema import MemoryRecord, FactType, Provenance, Tier

backend = get_backend()
store = MemoryStore(backend, db_path=":memory:")   # throwaway DB for the test


def days_ago(n):
    return datetime.now(timezone.utc) - timedelta(days=n)


veg = store.add(MemoryRecord(
    text="User is vegetarian and avoids crowded tourist spots.",
    fact_type=FactType.PREFERENCE, intents=["dining", "attractions"],
    confidence=0.95, provenance=Provenance(source="user")))

gion = store.add(MemoryRecord(
    text="Gion is a historic Kyoto district known for geisha culture and machiya townhouses.",
    fact_type=FactType.POI_FACT, subject="Kyoto", intents=["attractions", "accommodation"],
    confidence=0.5, provenance=Provenance(source="wikivoyage")))    # low conf -> starts WARM

visa_fresh = store.add(MemoryRecord(
    text="US citizens may enter Japan visa-free for up to 90 days.",
    fact_type=FactType.VISA_RULE, subject="Japan", intents=["entry_requirements"],
    confidence=0.85, observed_at=days_ago(3), provenance=Provenance(source="gov_advisory")))

visa_stale = store.add(MemoryRecord(
    text="Japan requires a pre-arrival quarantine declaration form.",
    fact_type=FactType.VISA_RULE, subject="Japan", intents=["entry_requirements"],
    confidence=0.85, observed_at=days_ago(200), provenance=Provenance(source="gov_advisory")))

print("=== after add: score + tier (fact_type half-life applied) ===")
for r in (veg, gion, visa_fresh, visa_stale):
    print(f"  {r.tier.value:5}  score={r.score:.3f}  {r.text[:50]}")
print("\nStale visa rule retired on ingest (note: raw score is non-trivial — "
      "the FRESHNESS FLOOR retires it, not the score):", visa_stale.tier is Tier.COLD)

print("\n=== search: 'good food options for me in Kyoto' ===")
for rec, dist in store.search("good food options for me in Kyoto", k=3):
    print(f"  dist={dist:.3f}  [{rec.tier.value}]  {rec.text[:50]}")

print("\n=== search: 'do I need a visa for Japan' (stale fact must NOT appear) ===")
for rec, dist in store.search("do I need a visa for Japan", k=5,
                              subject="Japan", fact_type=FactType.VISA_RULE):
    print(f"  dist={dist:.3f}  [{rec.tier.value}]  {rec.text[:50]}")

print("\n=== LOOP 2: repeated use promotes the Gion fact ===")
print(f"  before:  hits={gion.hit_count}  score={gion.score:.3f}  tier={gion.tier.value}")
for _ in range(5):
    store.record_access([gion])
print(f"  after 5: hits={gion.hit_count}  score={gion.score:.3f}  tier={gion.tier.value}")
print("  promoted WARM -> HOT:", gion.tier is Tier.HOT)

store.close()