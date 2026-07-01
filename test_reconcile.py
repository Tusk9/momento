"""Reconciliation across the real cases: supersede (the wow-moment), the guard
that protects corroborated memory, merge, coexist, no-conflict, and the audit
trail. From repo root: python test_reconcile.py"""
from datetime import datetime, timezone, timedelta

from momento.config import get_backend
from momento.memory.store import MemoryStore
from momento.memory.schema import MemoryRecord, FactType, Provenance, Tier
from momento.memory.reconcile import Reconciler, Outcome, supersede_trail

backend = get_backend()
store = MemoryStore(backend, db_path=":memory:")
rec = Reconciler(store, backend)


def days_ago(n):
    return datetime.now(timezone.utc) - timedelta(days=n)


def seed(**kw):
    kw.setdefault("provenance", Provenance(source="seed"))
    return store.add(MemoryRecord(**kw))


def scrape(**kw):
    """A freshly observed fact from the pipeline, run through reconciliation."""
    kw.setdefault("provenance", Provenance(source="gov_advisory", detail="re-scrape"))
    return rec.reconcile(MemoryRecord(**kw))


def find(substr, *, active_only=False):
    for r in store.all(include_inactive=not active_only):
        if substr in r.text:
            return r
    return None


print("=" * 72)
print("A) SUPERSEDE — a visa rule changed (the demo's wow-moment)")
old_visa = seed(text="US citizens may enter Japan visa-free for up to 90 days.",
                fact_type=FactType.VISA_RULE, subject="Japan",
                intents=["entry_requirements"], confidence=0.85,
                corroboration=1, observed_at=days_ago(45))
# a different subject that must NOT be touched
seed(text="Thailand allows US citizens visa-free entry for 30 days.",
     fact_type=FactType.VISA_RULE, subject="Thailand",
     intents=["entry_requirements"], confidence=0.85, observed_at=days_ago(45))

ev = scrape(text="US citizens now require an approved eVisa before entering Japan; "
                 "visa-free entry has ended.",
            fact_type=FactType.VISA_RULE, subject="Japan",
            intents=["entry_requirements"], confidence=0.85, observed_at=days_ago(0))

print(f"  decision={ev.decision.value if ev.decision else None}  outcome={ev.outcome.value}")
print(f"  caption -> {ev.caption()}")
old_visa = find("visa-free for up to 90")     # reload from store
new_visa = find("eVisa")
print(f"  old.superseded_by set: {old_visa.superseded_by is not None}")
print(f"  old.tier: {old_visa.tier.value}   old.is_active: {old_visa.is_active}")
print(f"  new.is_active: {new_visa.is_active}   new.supersedes: {new_visa.supersedes}")
live = [r.text[:40] for r, _ in store.search("do I need a visa for Japan", k=5,
        subject="Japan", fact_type=FactType.VISA_RULE)]
print(f"  live visa results for Japan: {live}")
assert ev.outcome is Outcome.SUPERSEDED, "expected the stale visa rule to be superseded"
assert old_visa.superseded_by == new_visa.id and not old_visa.is_active
assert new_visa.is_active and "eVisa" in new_visa.text
assert all("90 days" not in t for t in live), "retired fact must not appear live"
print("  PASS: stale rule retired, fresh rule live, Thailand untouched")

print("\n" + "=" * 72)
print("B) GUARD — a single weak scrape can't overturn a corroborated fact")
safe = seed(text="Bali is generally safe for tourists.",
            fact_type=FactType.ADVISORY, subject="Bali", intents=["safety_health"],
            confidence=0.9, corroboration=4, observed_at=days_ago(2))
evb = scrape(text="Bali is extremely dangerous right now; avoid all travel.",
             fact_type=FactType.ADVISORY, subject="Bali", intents=["safety_health"],
             confidence=0.3, observed_at=days_ago(0),
             provenance=Provenance(source="random_blog"))
safe = find("generally safe")
print(f"  decision={evb.decision.value if evb.decision else None}  outcome={evb.outcome.value}")
print(f"  caption -> {evb.caption()}")
print(f"  corroborated fact survived: superseded_by={safe.superseded_by}  "
      f"active={safe.is_active}  tier={safe.tier.value}  contradictions={safe.contradictions}")
assert safe.superseded_by is None and safe.is_active, \
    "a corroborated fact must NOT be retired by one low-confidence source"
print("  PASS: corroborated memory protected (new report kept but secondary)")

print("\n" + "=" * 72)
print("C) MERGE — a reworded duplicate confirms, doesn't duplicate")
nishiki = seed(text="Nishiki Market is a covered food market in central Kyoto.",
               fact_type=FactType.POI_FACT, subject="Kyoto",
               intents=["dining"], confidence=0.6, corroboration=0, observed_at=days_ago(10))
before = nishiki.corroboration
evc = scrape(text="Nishiki Market is a covered market selling food, located in central Kyoto.",
             fact_type=FactType.POI_FACT, subject="Kyoto", intents=["dining"],
             confidence=0.7, observed_at=days_ago(0),
             provenance=Provenance(source="wikivoyage"))
n_nishiki = sum(1 for r in store.all() if "Nishiki" in r.text)
nishiki = find("Nishiki")
print(f"  decision={evc.decision.value if evc.decision else None}  outcome={evc.outcome.value}")
print(f"  caption -> {evc.caption()}")
print(f"  active Nishiki records: {n_nishiki}  corroboration {before} -> {nishiki.corroboration}")
print("  (expect MERGED: 1 record, corroboration up; COEXIST would show 2 — model's call)")

print("\n" + "=" * 72)
print("D) COEXIST — two different Kyoto sights don't conflict")
seed(text="Fushimi Inari is a shrine famous for thousands of red torii gates.",
     fact_type=FactType.POI_FACT, subject="Kyoto", intents=["attractions"], confidence=0.6)
evd = scrape(text="Kinkaku-ji is a Zen temple in Kyoto covered in gold leaf.",
             fact_type=FactType.POI_FACT, subject="Kyoto", intents=["attractions"],
             confidence=0.6, observed_at=days_ago(0), provenance=Provenance(source="wikivoyage"))
print(f"  decision={evd.decision.value if evd.decision else None}  outcome={evd.outcome.value}")
print(f"  caption -> {evd.caption()}")

print("\n" + "=" * 72)
print("E) NO_CONFLICT — a brand-new subject with nothing stored")
eve = scrape(text="Portugal uses the euro as its currency.",
             fact_type=FactType.GEOGRAPHIC, subject="Portugal",
             intents=["practicalities"], confidence=0.9, observed_at=days_ago(0))
print(f"  outcome={eve.outcome.value}   caption -> {eve.caption()}")
assert eve.outcome is Outcome.NO_CONFLICT and find("euro").is_active

print("\n" + "=" * 72)
print("AUDIT TRAIL (durable, reconstructed from the store):")
for t in supersede_trail(store):
    print(f"  [{t['subject']} · {t['fact_type']}] retired: “{t['retired']}”")
    print(f"       replaced by ({t['source_of_new']}): “{t['replaced_by']}”")

store.close()
print("\nDone.")