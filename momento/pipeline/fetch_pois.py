"""Real data pipeline (thin slice): fetch Kyoto POIs from OpenStreetMap's
Overpass API and ingest them through the reconciler — the same guarded door
every fact enters by. License-clean (ODbL), no API key.

Run AFTER seed_demo.py, with the SAME backend as the server:
  MODEL_BACKEND=qwen_cloud python momento/pipeline/fetch_pois.py
"""
import os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import requests
from momento.config import get_backend, BACKEND
from momento.memory.store import MemoryStore
from momento.memory.schema import MemoryRecord, FactType, Provenance
from momento.memory.reconcile import Reconciler

DB = "data/demo.db"
OVERPASS = "https://overpass-api.de/api/interpreter"
MAX_POIS = 12   # keep the map readable and reconcile calls bounded

# Named tourist attractions & viewpoints in central Kyoto with English names
QUERY = """
[out:json][timeout:25];
(
  nwr["tourism"~"attraction|viewpoint"]["name:en"]["wikidata"](34.95,135.65,35.06,135.80);
);
out center %d;
""" % (MAX_POIS * 3)


def fetch_osm_pois() -> list[dict]:
    resp = requests.post(OVERPASS, data={"data": QUERY}, timeout=30,
    headers={"User-Agent": "momento-hackathon/0.1 (github.com/Tusk9/momento)"})
    resp.raise_for_status()
    out = []
    for el in resp.json().get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name:en")
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if not (name and lat and lon):
            continue
        kind = tags.get("tourism", "attraction")
        text = f"{name} is a notable {kind} in Kyoto."
        out.append({
            "text": text, "lat": lat, "lon": lon,
            "osm_url": f"https://www.openstreetmap.org/{el['type']}/{el['id']}",
        })
        if len(out) >= MAX_POIS:
            break
    return out


def main():
    assert os.path.exists(DB), "Run demo/seed_demo.py first (creates the DB)."
    backend = get_backend()
    store = MemoryStore(backend, db_path=DB)
    rec = Reconciler(store, backend)
    now = datetime.now(timezone.utc)

    pois = fetch_osm_pois()
    print(f"Fetched {len(pois)} POIs from OSM Overpass (backend: {BACKEND})\n")

    for p in pois:
        record = MemoryRecord(
            text=p["text"], fact_type=FactType.POI_FACT, subject="Kyoto",
            intents=["attractions", "itinerary", "trip_framing"],
            confidence=0.55,             # scraped single-source -> starts modest/WARM
            lat=p["lat"], lon=p["lon"], observed_at=now,
            provenance=Provenance(source="osm", detail=p["osm_url"]),
        )
        ev = rec.reconcile(record)
        print(f"  [{ev.outcome.value:12}] {p['text'][:60]}")

    n = len([r for r in store.all() if r.subject == 'Kyoto'])
    store.close()
    print(f"\nDone. {n} active Kyoto facts in {DB}.")


if __name__ == "__main__":
    main()