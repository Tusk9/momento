"""Verify a record builds, round-trips through dict, and infers its kind.
From repo root: python test_schema.py"""
from momento.memory.schema import MemoryRecord, FactType, MemoryKind, Tier, Provenance

# a stable user preference
pref = MemoryRecord(
    text="User is vegetarian.",
    fact_type=FactType.PREFERENCE,
    intents=["dining", "attractions"],
    confidence=0.95,
    provenance=Provenance(source="user", detail="stated in conversation"),
)

# a volatile scraped fact
visa = MemoryRecord(
    text="US citizens can enter Japan visa-free for up to 90 days.",
    fact_type=FactType.VISA_RULE,
    subject="Japan",
    intents=["entry_requirements"],
    confidence=0.8,
    provenance=Provenance(source="gov_advisory", detail="https://travel.state.gov/..."),
)

for r in (pref, visa):
    print(f"\n{r.id}  {r.fact_type.value}")
    print(f"  kind inferred: {r.kind.value}   volatile={r.is_volatile}   active={r.is_active}")
    # round-trip
    clone = MemoryRecord.from_dict(r.to_dict())
    assert clone.to_dict() == r.to_dict(), "round-trip mismatch!"
    print("  round-trip: OK")

print("\nKind inference correct:",
      pref.kind is MemoryKind.STABLE and visa.kind is MemoryKind.VOLATILE)