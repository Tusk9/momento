"""The agent loop — the orchestrator that turns the parts into momento.

Per user turn:
  1. INGEST fresh facts (optional): run any freshly-scraped/injected facts
     through the reconciler FIRST, so this turn reasons on UPDATED memory
     (the session-3 'a rule changed' beat happens here, before the answer).
  2. CLASSIFY the turn -> intent, entities, horizon (fast model).
  3. PREFETCH (Loop 1): warm a hot buffer from current + predicted intents,
     measuring prediction_hit against last turn's prediction.
  4. FALLBACK: reactive vector search if the prefetch missed the current intent.
  5. REASON: main model answers, grounded on the hot buffer.
  6. PROMOTE (Loop 2): record_access on the memories that grounded THIS turn
     (current-intent hits / fallback hits). Warming alone never promotes.
  7. WRITE BACK: on preference-ish turns, extract durable user facts and route
     them through the reconciler too (one unified, guarded ingest path).
  8. CARRY intent + prediction + subject forward for the next turn.

Model calls per ORDINARY turn: classify (fast) + reason (main). Extraction and
reconciliation fire only when warranted (a stated preference, or a real conflict).
Long conversation history is intentionally trimmed — persistent memory, not a
growing transcript, is what carries context across turns. That's the thesis.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

from momento.models.base import LLMBackend
from momento.intent.taxonomy import Intent
from momento.intent.classifier import classify, Classification
from momento.memory.schema import MemoryRecord, FactType, Provenance
from momento.memory.store import MemoryStore
from momento.memory.prefetch import Prefetcher, PrefetchResult, render_buffer
from momento.memory.reconcile import Reconciler, ReconciliationEvent

REASONING_SYSTEM = (
    "You are momento, a travel-planning assistant with long-term memory across "
    "sessions. When memory about the user or a destination is provided, use it to "
    "personalize and stay consistent with what you've learned. Prefer the user's "
    "latest statements over stored memory if they conflict, and never invent facts "
    "you don't have. Be concise, concrete, and practical."
    "Keep answers under ~150 words unless the user asks for detail."
)

# Extraction (step 7) runs when the turn states something durable about the user.
DURABLE_ENTITY_TYPES = {"cuisine", "companion", "budget"}
_EXTRACT_FACT_TYPES = {"preference": FactType.PREFERENCE,
                       "user_profile": FactType.USER_PROFILE}
_VALID_INTENTS = {i.value for i in Intent}
# fallback tags so a learned preference still surfaces if extraction omits intents
_DEFAULT_PREF_INTENTS = ["dining", "attractions", "accommodation", "itinerary", "transport"]

_HISTORY_MSGS = 8   # last N chat messages sent to the model (memory carries the rest)


def _subject_from(c: Classification) -> str | None:
    for t in ("place", "poi"):
        for e in c.entities:
            if e.get("type") == t:
                return e.get("value")
    return None


def _strip_think(s: str) -> str:
    """Some reasoning models emit <think>…</think>; keep it out of the reply."""
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()


def _extract_prompt(text: str) -> list[dict]:
    intent_names = ", ".join(i.value for i in Intent)
    system = (
        "Extract DURABLE facts about the USER from their message — lasting "
        "preferences, constraints, or profile traits still true on a future trip "
        "(dietary needs, travel style, pace, budget band, mobility, companions, hard "
        "likes/dislikes). Do NOT extract trip-specific or one-off wishes (a place they "
        "are currently considering, a single request).\n"
        "For each fact provide:\n"
        "  text: a short third-person statement about the user\n"
        "  type: \"preference\" (a like/dislike/constraint) or \"user_profile\" (a stable trait)\n"
        f"  intents: which planning topics it is relevant to, from: {intent_names}\n"
        "Respond with ONLY JSON:\n"
        '{"facts": [{"text": "...", "type": "preference", "intents": ["dining"]}]}\n'
        'If there are no durable user facts, return {"facts": []}.'
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": text}]


@dataclass
class AgentTurn:
    """Everything that happened in one turn — the observability surface the demo
    and tests read from."""
    reply: str
    classification: Classification
    prefetch: PrefetchResult
    used_fallback: bool
    grounding: list[MemoryRecord]                                   # memories shown to the model
    promoted: list[MemoryRecord]                                    # bumped this turn (Loop 2)
    learned: list[MemoryRecord] = field(default_factory=list)       # new durable user facts
    ingest_events: list[ReconciliationEvent] = field(default_factory=list)  # fresh + extracted
    subject: str | None = None

    def summary(self) -> str:
        pr = self.prefetch
        return (
            f"intent={self.classification.intent.value}  subject={self.subject or '—'}\n"
            f"  prefetch: {len(pr.buffer)} mem ~{pr.token_estimate}tok  "
            f"intent_hit={pr.intent_hit}  prediction_hit={pr.prediction_hit}  "
            f"fallback={self.used_fallback}\n"
            f"  promoted={len(self.promoted)}  learned={len(self.learned)}  "
            f"ingest_events={len(self.ingest_events)}"
        )


class Agent:
    def __init__(self, store: MemoryStore, backend: LLMBackend, *,
                 user_id: str = "default", token_budget: int = 400):
        self.store = store
        self.backend = backend
        self.user_id = user_id
        self.prefetcher = Prefetcher(store, token_budget=token_budget)
        self.reconciler = Reconciler(store, backend)
        self.history: list[dict] = []
        self._recent_intents: list[Intent] = []
        self._last_predicted: list[Intent] | None = None
        self._current_subject: str | None = None

    def new_session(self) -> None:
        """Start a fresh conversation. Clears transcript/turn state but KEEPS the
        memory store — this is how momento 'remembers you' across sessions."""
        self.history.clear()
        self._recent_intents.clear()
        self._last_predicted = None
        self._current_subject = None

    # --- one turn ------------------------------------------------------
    def turn(self, text: str, *, fresh_facts: list[MemoryRecord] | None = None) -> AgentTurn:
        ingest_events: list[ReconciliationEvent] = []

        # 1. INGEST fresh facts first -> reason on updated memory.
        for fact in (fresh_facts or []):
            fact.user_id = self.user_id
            ingest_events.append(self.reconciler.reconcile(fact, user_id=self.user_id))

        # 2. CLASSIFY (recent intents help disambiguate short follow-ups).
        c = classify(text, self.backend, recent_intents=self._recent_intents)
        subject = _subject_from(c) or self._current_subject   # carry subject if none this turn

        # 3. PREFETCH (Loop 1).
        pr = self.prefetcher.prefetch(
            c, query=text, subject=subject, user_id=self.user_id,
            predicted_last_turn=self._last_predicted,
        )

        # 4. FALLBACK on a current-intent miss (reactive, no correctness cost).
        used_fallback = False
        grounding = list(pr.buffer)
        fallback_hits: list[MemoryRecord] = []
        if pr.needs_fallback:
            used_fallback = True
            fallback_hits = [r for r, _d in
                             self.prefetcher.fallback_search(text, user_id=self.user_id, k=5)]
            seen = {r.id for r in grounding}
            for r in fallback_hits:
                if r.id not in seen:
                    grounding.append(r); seen.add(r.id)

        # 5. REASON, grounded on the hot buffer.
        reply = self._reason(text, grounding)

        # 6. PROMOTE (Loop 2) — only what actually grounded this turn.
        promoted = self._promote(grounding, c.intent, fallback_hits)

        # 7. WRITE BACK durable user facts (gated).
        learned: list[MemoryRecord] = []
        if self._should_extract(c):
            learned, extract_events = self._extract_and_store(text)
            ingest_events.extend(extract_events)

        # 8. CARRY forward.
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": reply}]
        self._recent_intents = (self._recent_intents + [c.intent])[-5:]
        self._last_predicted = list(pr.predicted_intents)
        if _subject_from(c):
            self._current_subject = _subject_from(c)

        return AgentTurn(reply=reply, classification=c, prefetch=pr,
                         used_fallback=used_fallback, grounding=grounding,
                         promoted=promoted, learned=learned,
                         ingest_events=ingest_events, subject=subject)

    # --- helpers -------------------------------------------------------
    def _reason(self, text: str, grounding: list[MemoryRecord]) -> str:
        memory_block = render_buffer(grounding)
        user_content = text if not memory_block else f"{memory_block}\n\n---\nUser: {text}"
        messages = ([{"role": "system", "content": REASONING_SYSTEM}]
                    + self.history[-_HISTORY_MSGS:]
                    + [{"role": "user", "content": user_content}])
        return _strip_think(self.backend.chat(messages, temperature=0.4, max_tokens=800))

    def _promote(self, grounding: list[MemoryRecord], current_intent: Intent,
                 fallback_hits: list[MemoryRecord]) -> list[MemoryRecord]:
        """Promote memories that grounded THIS answer: those carrying the current
        intent, plus fallback hits (relevance-matched to the query). Predicted-only
        warms are NOT promoted — warming isn't using."""
        fallback_ids = {r.id for r in fallback_hits}
        to_promote, ids = [], set()
        for r in grounding:
            if (current_intent.value in r.intents or r.id in fallback_ids) and r.id not in ids:
                to_promote.append(r); ids.add(r.id)
        if to_promote:
            self.store.record_access(to_promote)
        return to_promote

    def _should_extract(self, c: Classification) -> bool:
        return (c.intent is Intent.PREFERENCE
                or any(e.get("type") in DURABLE_ENTITY_TYPES for e in c.entities))

    def _extract_and_store(self, text: str) -> tuple[list[MemoryRecord], list[ReconciliationEvent]]:
        learned: list[MemoryRecord] = []
        events: list[ReconciliationEvent] = []
        try:
            data = self.backend.complete_json(_extract_prompt(text), fast=False, temperature=0.0)
        except Exception as e:
            print(f"[extract] FAILED to parse durable-fact JSON: {e}")
            return learned, events
        for f in data.get("facts", []) or []:
            ftype = _EXTRACT_FACT_TYPES.get(str(f.get("type", "")).strip().lower())
            body = str(f.get("text", "")).strip()
            if not ftype or not body:
                continue
            intents = [x for x in (f.get("intents") or [])
                       if isinstance(x, str) and x in _VALID_INTENTS] or _DEFAULT_PREF_INTENTS
            rec = MemoryRecord(
                text=body, fact_type=ftype, user_id=self.user_id, intents=intents,
                confidence=0.9,
                provenance=Provenance(source="user", detail="stated in conversation"),
            )
            events.append(self.reconciler.reconcile(rec, user_id=self.user_id))  # unified ingest
            learned.append(rec)
        return learned, events