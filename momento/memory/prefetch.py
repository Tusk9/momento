"""Loop 1 — predictive prefetch.

Before the agent reasons, warm a small hot buffer with the memories most likely
to matter THIS turn and NEXT turn:

  - current intent  -> relevance: vector-search the user's ACTUAL message
  - predicted intents (from transitions.predict_next) -> score: we have no query
    yet, so pull the highest-scored memories carrying those intent tags
  - always-on HOT tier -> the persistent prefs/profile injected every session

Pack by priority into a ~token_budget buffer (HOT, then current, then predicted),
deduped, stopping when the budget is hit. Report whether we found anything for the
current intent (intent_hit) so the agent can fall back to a reactive broad search
on a miss — a miss costs a wasted prefetch, never correctness.

Promotion (Loop 2) is NOT done here by default: warming != using. The agent
promotes the memories it actually grounds its answer in via store.record_access().
The promote=True flag promotes the whole buffer, for demos/tests only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone

from momento.intent.taxonomy import Intent
from momento.intent.classifier import Classification
from momento.intent import transitions
from momento.memory.schema import MemoryRecord, Tier
from momento.memory import scoring
from momento.memory.store import MemoryStore


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token). Approximate — real tokenization is
    model-specific — but good enough to watch the buffer fill and tune budget."""
    return max(1, (len(text) + 3) // 4)


def render_memory(r: MemoryRecord) -> str:
    subj = f" · {r.subject}" if r.subject else ""
    return f"- ({r.fact_type.value}{subj}) {r.text}"


def render_buffer(records: list[MemoryRecord]) -> str:
    if not records:
        return ""
    body = "\n".join(render_memory(r) for r in records)
    return "Relevant memory (prefetched for this turn):\n" + body


@dataclass
class PrefetchResult:
    buffer: list[MemoryRecord]
    current_intent: Intent
    predicted_intents: list[Intent]
    token_estimate: int
    intent_hit: bool                       # did we find memories for the CURRENT intent?
    prediction_hit: bool | None = None     # was this intent anticipated last turn? (Loop-1 metric)
    counts: dict = field(default_factory=dict)  # {hot, current, predicted} contributions

    def text(self) -> str:
        return render_buffer(self.buffer)

    @property
    def needs_fallback(self) -> bool:
        return not self.intent_hit

    def summary(self) -> str:
        preds = ", ".join(i.value for i in self.predicted_intents) or "—"
        pred = {True: "yes", False: "no", None: "n/a"}[self.prediction_hit]
        return (f"intent={self.current_intent.value}  predicted=[{preds}]\n"
                f"buffer={len(self.buffer)} memories  ~{self.token_estimate} tok"
                f"  (hot={self.counts.get('hot',0)} current={self.counts.get('current',0)}"
                f" predicted={self.counts.get('predicted',0)})\n"
                f"intent_hit={self.intent_hit}  prediction_hit={pred}"
                f"  needs_fallback={self.needs_fallback}")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _subject_from(c: Classification) -> str | None:
    """Best subject guess from the turn's entities: a place beats a POI."""
    for t in ("place", "poi"):
        for e in c.entities:
            if e.get("type") == t:
                return e.get("value")
    return None


class Prefetcher:
    def __init__(self, store: MemoryStore, *, token_budget: int = 400,
                 n_predicted: int = 2, per_intent: int = 3, max_hot: int = 4):
        self.store = store
        self.token_budget = token_budget
        self.n_predicted = n_predicted      # how many predicted intents to warm
        self.per_intent = per_intent        # memories per intent
        self.max_hot = max_hot              # cap on always-on HOT injections

    # --- selection helpers --------------------------------------------
    def _scored(self, records: list[MemoryRecord], now: datetime) -> list[MemoryRecord]:
        """Rank by FRESH score (recompute, don't trust the cached value — a
        volatile fact may have decayed since it was last touched)."""
        return sorted(records, key=lambda r: scoring.score(r, now), reverse=True)

    def _subject_ok(self, r: MemoryRecord, subject: str | None) -> bool:
        # global (user) facts always apply; subject-scoped facts must match.
        return subject is None or r.subject is None or r.subject == subject

    def _candidates(self, records: list[MemoryRecord], intent: Intent,
                    subject: str | None, now: datetime) -> list[MemoryRecord]:
        return [r for r in records
                if intent.value in r.intents
                and not scoring.is_stale(r, now)
                and self._subject_ok(r, subject)]

    # --- main ----------------------------------------------------------
    def prefetch(self, classification: Classification, *, query: str | None = None,
                 subject: str | None = None, user_id: str = "default",
                 predicted_last_turn: list[Intent] | None = None,
                 promote: bool = False) -> PrefetchResult:
        now = _now()
        current = classification.intent
        if subject is None:
            subject = _subject_from(classification)

        # next-intent predictions, dropping the current intent if present
        ranked = transitions.predict_next(current, top_n=self.n_predicted + 1)
        predicted = [i for i, _p in ranked if i != current][:self.n_predicted]

        all_recs = [r for r in self.store.all() if r.user_id == user_id]

        # 1) always-on HOT (persistent prefs/profile), score-ranked
        hot = self._scored(
            [r for r in all_recs if r.tier is Tier.HOT
             and not scoring.is_stale(r, now) and self._subject_ok(r, subject)],
            now,
        )

        # 2) current intent — relevance via the user's real message if we have it
        if query:
            hits = self.store.search(query, k=self.per_intent * 3,
                                     user_id=user_id, intent=current.value)
            current_relevant = [r for r, _d in hits if self._subject_ok(r, subject)]
        else:
            current_relevant = self._scored(
                self._candidates(all_recs, current, subject, now), now)
        current_relevant = current_relevant[:self.per_intent]

        # 3) predicted intents — score-ranked (no query to be relevant against)
        predicted_pulls: list[MemoryRecord] = []
        for pi in predicted:
            predicted_pulls += self._scored(
                self._candidates(all_recs, pi, subject, now), now)[:self.per_intent]

        # assemble by priority into the budget, deduped
        buffer: list[MemoryRecord] = []
        seen: set[str] = set()
        counts = {"hot": 0, "current": 0, "predicted": 0}
        running = estimate_tokens("Relevant memory (prefetched for this turn):")

        def add(records, bucket, cap):
            nonlocal running
            n = 0
            for r in records:
                if r.id in seen:
                    continue
                cost = estimate_tokens(render_memory(r))
                if running + cost > self.token_budget:
                    return                       # budget hit — stop (priority order respected)
                seen.add(r.id)
                buffer.append(r)
                running += cost
                counts[bucket] += 1
                n += 1
                if n >= cap:
                    return

        add(hot, "hot", self.max_hot)
        add(current_relevant, "current", self.per_intent)
        add(predicted_pulls, "predicted", self.per_intent * self.n_predicted)

        if promote and buffer:
            self.store.record_access(buffer)     # demo/test only; agent promotes on USE

        prediction_hit = None
        if predicted_last_turn is not None:
            prediction_hit = current in predicted_last_turn

        return PrefetchResult(
            buffer=buffer, current_intent=current, predicted_intents=predicted,
            token_estimate=running, intent_hit=len(current_relevant) > 0,
            prediction_hit=prediction_hit, counts=counts,
        )

    def fallback_search(self, query: str, *, user_id: str = "default",
                        k: int = 5) -> list[tuple[MemoryRecord, float]]:
        """Reactive path for a prefetch miss: broad vector search, no intent
        filter. The agent calls this when result.needs_fallback is True."""
        return self.store.search(query, k=k, user_id=user_id)