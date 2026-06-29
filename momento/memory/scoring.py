"""Scoring + freshness — the 'forgets on purpose' core.

Two jobs, kept separate:
  1. score(record): a single ranking number the prefetcher and tiering use.
     Additive, per the design:
        w_f*freshness + w_t*confidence + w_u*usefulness + w_c*corroboration
        + w_r*recency - w_p*contradiction
  2. is_stale(record): for VOLATILE facts only, whether freshness has decayed
     past a floor -> PASSIVE retirement (a fact rots on its own).
     ACTIVE retirement (a new fact contradicts an old one) lives in reconcile.py.

Per-fact-type half-lives live in config.HALF_LIVES_DAYS (one tunable table).
Weights + thresholds are scoring-internal and live here.
"""
from __future__ import annotations
import math
from datetime import datetime, timezone

from momento.memory.schema import MemoryRecord, MemoryKind, Tier
from momento.config import HALF_LIVES_DAYS

# --- weights (positive terms sum to 1.0 -> score is interpretable) ---
W_FRESHNESS     = 0.30
W_TRUST         = 0.25   # confidence
W_USEFULNESS    = 0.20   # hit_count (LOOP 2)
W_CORROBORATION = 0.15
W_RECENCY       = 0.10   # time since last use (LOOP 2)
W_CONTRADICTION = 0.30   # penalty (subtracted)

# saturation constants: how fast a count's contribution approaches 1
_USE_K    = 3.0          # ~3 hits ≈ 0.64 usefulness
_CORROB_K = 2.0
_CONTRA_K = 2.0
_RECENCY_HALFLIFE_DAYS = 30.0

# tier thresholds on the score
HOT_THRESHOLD  = 0.55
WARM_THRESHOLD = 0.25

# below this freshness, a VOLATILE fact is considered stale -> retire
RETIRE_FRESHNESS = 0.15


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _age_days(ts: datetime, now: datetime) -> float:
    return max(0.0, (now - ts).total_seconds() / 86400.0)


def freshness(record: MemoryRecord, now: datetime | None = None) -> float:
    """How current the underlying real-world fact is, 0..1. Half-life decay on
    observed_at, with the half-life chosen by fact_type. STABLE types have huge
    half-lives, so this stays ~1 for them."""
    now = now or _now()
    half_life = HALF_LIVES_DAYS[record.fact_type]
    return 0.5 ** (_age_days(record.observed_at, now) / half_life)


def _saturating(count: int, k: float) -> float:
    return 1.0 - math.exp(-count / k)


def _recency(record: MemoryRecord, now: datetime) -> float:
    """Recently-used (or newly-learned) memories rank higher. Gentle decay."""
    ref = record.last_accessed or record.created_at
    return 0.5 ** (_age_days(ref, now) / _RECENCY_HALFLIFE_DAYS)


def score(record: MemoryRecord, now: datetime | None = None) -> float:
    """The single ranking number. Higher = more worth surfacing."""
    now = now or _now()
    f = freshness(record, now)
    u = _saturating(record.hit_count, _USE_K)
    c = _saturating(record.corroboration, _CORROB_K)
    p = _saturating(record.contradictions, _CONTRA_K)
    r = _recency(record, now)
    return (
        W_FRESHNESS * f
        + W_TRUST * record.confidence
        + W_USEFULNESS * u
        + W_CORROBORATION * c
        + W_RECENCY * r
        - W_CONTRADICTION * p
    )


def tier_for(score_value: float) -> Tier:
    if score_value >= HOT_THRESHOLD:
        return Tier.HOT
    if score_value >= WARM_THRESHOLD:
        return Tier.WARM
    return Tier.COLD


def is_stale(record: MemoryRecord, now: datetime | None = None) -> bool:
    """Passive retirement: a VOLATILE fact whose freshness fell past the floor.
    Stable facts never go stale this way."""
    if record.kind is not MemoryKind.VOLATILE:
        return False
    return freshness(record, now) < RETIRE_FRESHNESS


def rescore(record: MemoryRecord, now: datetime | None = None) -> MemoryRecord:
    """Recompute score + tier in place. A stale volatile fact is forced COLD
    (retired) regardless of its raw score. Returns the record."""
    now = now or _now()
    record.score = score(record, now)
    record.tier = Tier.COLD if is_stale(record, now) else tier_for(record.score)
    return record