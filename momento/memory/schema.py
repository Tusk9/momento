"""The MemoryRecord: one fact the agent knows, plus all the metadata the two
loops and the contradiction reconciler need.

Field-to-purpose map (why each exists):
  - embedding/text ........ retrieval (vector search finds candidates)
  - kind/fact_type ........ STABLE vs VOLATILE split; picks the freshness half-life
  - tier .................. hot/warm/cold (what gets injected vs retrieved vs archived)
  - score ................. cached ranking output (computed in scoring.py)
  - hit_count/last_accessed recency_decay + usefulness  -> LOOP 2 (use promotes)
  - confidence/corroboration/contradictions ... trust signals in the score
  - intents ............... LOOP 1: which predicted intents this memory serves
  - created_at/observed_at  age, and when the underlying fact was true
  - superseded_by/supersedes  contradiction reconciliation provenance trail
  - provenance ............ where the fact came from (auditable, judge-visible)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone
import uuid


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class MemoryKind(str, Enum):
    """The split your whole design exploits."""
    STABLE = "stable"      # accumulates, gets promoted (preferences, places visited)
    VOLATILE = "volatile"  # must be retired when it goes stale (hours, visas, prices)


class Tier(str, Enum):
    HOT = "hot"      # injected into every session
    WARM = "warm"    # retrieved on demand via vector search
    COLD = "cold"    # archived; not retrieved, but auditable


class FactType(str, Enum):
    """Drives the freshness half-life. STABLE types decay slowly or never;
    VOLATILE types decay fast. Half-lives live in config.py (one tunable table),
    NOT here — the schema only names the type."""
    # stable
    PREFERENCE = "preference"          # vegetarian, hates early flights
    USER_PROFILE = "user_profile"      # budget band, pace, mobility, companions
    VISITED = "visited"                # places the user has been
    GEOGRAPHIC = "geographic"          # "Kyoto is in Japan" — effectively never stale
    # volatile
    VISA_RULE = "visa_rule"            # entry requirements (weeks)
    HOURS = "hours"                    # opening hours (days)
    PRICE = "price"                    # fares, ticket costs (days)
    CLOSURE = "closure"                # seasonal/temporary closures (days–weeks)
    TRANSIT = "transit"                # route/schedule changes (days–weeks)
    ADVISORY = "advisory"             # safety/weather/health advisories (days)
    POI_FACT = "poi_fact"             # general fact about a place (slow)


# Which kind each fact type belongs to. Used to set defaults and to validate.
FACT_KIND: dict[FactType, MemoryKind] = {
    FactType.PREFERENCE: MemoryKind.STABLE,
    FactType.USER_PROFILE: MemoryKind.STABLE,
    FactType.VISITED: MemoryKind.STABLE,
    FactType.GEOGRAPHIC: MemoryKind.STABLE,
    FactType.POI_FACT: MemoryKind.STABLE,
    FactType.VISA_RULE: MemoryKind.VOLATILE,
    FactType.HOURS: MemoryKind.VOLATILE,
    FactType.PRICE: MemoryKind.VOLATILE,
    FactType.CLOSURE: MemoryKind.VOLATILE,
    FactType.TRANSIT: MemoryKind.VOLATILE,
    FactType.ADVISORY: MemoryKind.VOLATILE,
}


@dataclass
class Provenance:
    """Where a fact came from — auditable, and shown on screen in the demo
    when a stale fact gets retired."""
    source: str                          # "user" | "osm" | "gov_advisory" | "wikivoyage" | ...
    detail: str = ""                     # URL, dataset name, or "stated in conversation"
    fetched_at: datetime = field(default_factory=_now)


@dataclass
class MemoryRecord:
    text: str                                   # the fact, in natural language
    fact_type: FactType
    kind: MemoryKind = None                      # auto-filled from fact_type if omitted

    # identity + place scoping
    id: str = field(default_factory=_new_id)
    subject: str | None = None                   # the place/entity this is about ("Kyoto"); None = about the user
    user_id: str = "default"                     # multi-user ready, single user for the demo
    lat: float | None = None                     # map coordinates, for POI-type facts
    lon: float | None = None
    
    # retrieval
    embedding: list[float] | None = None         # filled by the store on add()

    # LOOP 1 — prediction routing
    intents: list[str] = field(default_factory=list)  # which Intent values this memory serves

    # LOOP 2 — use promotes
    hit_count: int = 0
    last_accessed: datetime | None = None

    # trust signals (feed the score)
    confidence: float = 0.7                      # 0–1; user-stated facts start high, scraped lower
    corroboration: int = 0                       # independent sources/confirmations
    contradictions: int = 0                      # times a conflicting fact was seen

    # tiering + cached ranking
    tier: Tier = Tier.WARM
    score: float = 0.0                           # recomputed by scoring.py

    # time
    created_at: datetime = field(default_factory=_now)   # when WE recorded it
    observed_at: datetime = field(default_factory=_now)  # when the fact was actually true/sourced

    # contradiction reconciliation trail
    superseded_by: str | None = None             # id of the record that replaced this one
    supersedes: list[str] = field(default_factory=list)  # ids this record replaced

    provenance: Provenance | None = None

    def __post_init__(self):
        if self.kind is None:
            self.kind = FACT_KIND.get(self.fact_type, MemoryKind.STABLE)

    @property
    def is_active(self) -> bool:
        """A superseded or cold-archived fact is never injected/retrieved for use."""
        return self.superseded_by is None and self.tier is not Tier.COLD

    @property
    def is_volatile(self) -> bool:
        return self.kind is MemoryKind.VOLATILE

    def to_dict(self) -> dict:
        """Flatten for storage. Enums -> str, datetimes -> ISO strings."""
        d = asdict(self)
        for k in ("fact_type", "kind", "tier"):
            d[k] = getattr(self, k).value
        for k in ("created_at", "observed_at", "last_accessed"):
            v = getattr(self, k)
            d[k] = v.isoformat() if v else None
        if self.provenance:
            d["provenance"] = {
                "source": self.provenance.source,
                "detail": self.provenance.detail,
                "fetched_at": self.provenance.fetched_at.isoformat(),
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        d = dict(d)
        d["fact_type"] = FactType(d["fact_type"])
        d["kind"] = MemoryKind(d["kind"])
        d["tier"] = Tier(d["tier"])
        for k in ("created_at", "observed_at", "last_accessed"):
            d[k] = datetime.fromisoformat(d[k]) if d.get(k) else None
        prov = d.get("provenance")
        if prov:
            d["provenance"] = Provenance(
                source=prov["source"], detail=prov.get("detail", ""),
                fetched_at=datetime.fromisoformat(prov["fetched_at"]),
            )
        return cls(**d)