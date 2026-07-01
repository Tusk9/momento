"""End-to-end: a short multi-session conversation exercising the whole loop —
learn a preference, survive a session reset, prefetch it on return, and catch a
real-world fact change mid-conversation. From repo root: python test_agent.py

Note: this runs several REAL model generations locally — expect ~30–60s.
"""
from datetime import datetime, timezone, timedelta

from momento.config import get_backend
from momento.memory.store import MemoryStore
from momento.memory.schema import MemoryRecord, FactType, Provenance, Tier
from momento.memory.reconcile import Outcome
from momento.agent import Agent

backend = get_backend()
store = MemoryStore(backend, db_path=":memory:")
agent = Agent(store, backend)


def days_ago(n):
    return datetime.now(timezone.utc) - timedelta(days=n)


def show(label, t):
    print("\n" + "=" * 72)
    print(label)
    print("-" * 72)
    print("momento:", (t.reply[:280] + "…") if len(t.reply) > 280 else t.reply)
    print("-" * 72)
    print(t.summary())


def prefs_in_store():
    return [r for r in store.all() if r.fact_type is FactType.PREFERENCE]


# A visa rule 'learned in an earlier session' — the fact that will later change.
store.add(MemoryRecord(
    text="US citizens may enter Japan visa-free for up to 90 days.",
    fact_type=FactType.VISA_RULE, subject="Japan", intents=["entry_requirements"],
    confidence=0.85, corroboration=1, observed_at=days_ago(45),
    provenance=Provenance(source="gov_advisory")))

print("###  SESSION 1  — plan + learn the user  ###")

tA = agent.turn("We're planning a week in Kyoto in October.")
show("TURN A — frame the trip", tA)

tB = agent.turn("I'm vegetarian and I really dislike crowded tourist spots — "
                "please remember that for planning.")
show("TURN B — state a durable preference", tB)
assert len(tB.learned) >= 1, "expected a durable preference to be extracted"
pref = next((p for p in prefs_in_store()), None)
assert pref is not None and pref.is_active, "preference should be stored + active"
print("\n  ✓ learned + stored preference:", f'“{pref.text}”  tier={pref.tier.value}')
assert pref.tier is Tier.HOT, "a fresh high-confidence preference should land HOT"
print("  ✓ preference is HOT (will be injected every future session)")

# --- the user leaves and comes back later: fresh conversation, same memory ---
agent.new_session()
print("\n\n###  SESSION 2  — the user returns (transcript reset, memory persists)  ###")

tD = agent.turn("Hey, I'm back — planning that Kyoto trip. Where should we grab dinner?")
show("TURN D — return + a dining question", tD)
buf_types = {r.fact_type for r in tD.prefetch.buffer}
assert FactType.PREFERENCE in buf_types, \
    "the vegetarian preference should be PREFETCHED across the session boundary"
print("\n  ✓ preference prefetched in a brand-new session (cross-session memory works)")
assert len(tD.promoted) >= 1, "a used memory should be promoted (Loop 2)"
print("  ✓ used memory promoted (Loop 1 -> Loop 2)")

print("\n\n###  SESSION 3  — a real-world fact changed  ###")

# The pipeline (simulated here) hands the agent a freshly-scraped fact that
# conflicts with what we stored 45 days ago.
new_visa = MemoryRecord(
    text="US citizens now need an approved eVisa before entering Japan; "
         "visa-free entry has ended.",
    fact_type=FactType.VISA_RULE, subject="Japan", intents=["entry_requirements"],
    confidence=0.85, observed_at=days_ago(0),
    provenance=Provenance(source="gov_advisory", detail="re-scrape 2026"))

tE = agent.turn("Do I still need a visa for Japan?", fresh_facts=[new_visa])
show("TURN E — the fact-change beat (reconcile fires before the answer)", tE)

superseded = [e for e in tE.ingest_events if e.outcome is Outcome.SUPERSEDED]
assert superseded, "the stale visa rule should have been superseded"
print("\n  ✓ reconciliation retired the stale rule:")
print("     " + superseded[0].caption())

stale = next(r for r in store.all(include_inactive=True) if "90 days" in r.text)
fresh = next(r for r in store.all() if "eVisa" in r.text)
assert not stale.is_active and fresh.is_active, "stale retired, fresh live"
ground_txt = " ".join(r.text for r in tE.grounding)
assert "eVisa" in ground_txt and "90 days" not in ground_txt, \
    "the answer should be grounded on the FRESH rule, not the retired one"
print("  ✓ this turn reasoned on the updated fact (fresh in grounding, stale gone)")

print("\n" + "=" * 72)
print("ALL CORE BEHAVIORS VERIFIED — momento is wired end to end.")
store.close()