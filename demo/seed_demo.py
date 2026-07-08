"""Build the demo database: Kyoto POIs (with coords), one stale-ish visa rule,
and provenance everywhere. Run with the SAME backend you'll serve with:
  local:  python demo/seed_demo.py
  cloud:  MODEL_BACKEND=qwen_cloud python demo/seed_demo.py
(Embeddings from different models live in different vector spaces — never mix.)
"""
import os, sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momento.config import get_backend, BACKEND
from momento.memory.store import MemoryStore
from momento.memory.schema import MemoryRecord, FactType, Provenance

DB = "data/demo.db"

if os.path.exists(DB):
    os.remove(DB)

backend = get_backend()
store = MemoryStore(backend, db_path=DB)
now = datetime.now(timezone.utc)

def days_ago(n): return now - timedelta(days=n)

WIKIVOYAGE = "https://en.wikivoyage.org/wiki/Kyoto"
ADVISORY = "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Japan.html"

POIS = [
    ("Nishiki Market is a covered food street in central Kyoto with 100+ stalls.",
     35.0050, 135.7649, ["dining", "attractions", "trip_framing"]),
    ("Gion is Kyoto's historic geisha district with preserved machiya townhouses.",
     35.0037, 135.7752, ["attractions", "accommodation","trip_framing"]),
    ("Fushimi Inari Taisha is famous for thousands of vermilion torii gates.",
     34.9671, 135.7727, ["attractions", "itinerary","trip_framing"]),
    ("Kinkaku-ji (Golden Pavilion) is a Zen temple covered in gold leaf.",
     35.0394, 135.7292, ["attractions", "itinerary","trip_framing"]),
    ("Arashiyama has a famous bamboo grove and riverside walks, best early morning.",
     35.0094, 135.6668, ["attractions", "itinerary","trip_framing"]),
    ("Downtown Kawaramachi is central and well-connected — a practical base to stay.",
     35.0031, 135.7686, ["accommodation","trip_framing"]),
]

for text, lat, lon, intents in POIS:
    store.add(MemoryRecord(
        text=text, fact_type=FactType.POI_FACT, subject="Kyoto",
        intents=intents, confidence=0.6, lat=lat, lon=lon,
        observed_at=days_ago(10),
        provenance=Provenance(source="wikivoyage", detail=WIKIVOYAGE)))

# The fact that will change in session 3 (stored 45 days ago, still fresh enough to be live)
store.add(MemoryRecord(
    text="US citizens may enter Japan visa-free for stays up to 90 days.",
    fact_type=FactType.VISA_RULE, subject="Japan", intents=["entry_requirements"],
    confidence=0.85, corroboration=1, observed_at=days_ago(45),
    provenance=Provenance(source="gov_advisory", detail=ADVISORY)))

n = len(store.all())
store.close()
print(f"Seeded {DB} with {n} memories using backend '{BACKEND}'.")