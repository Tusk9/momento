"""Contradiction reconciliation — the active-retirement path.

When a freshly scraped fact arrives, decide its relationship to what we already
know about the SAME subject + category:

  - supersede : same aspect, DIFFERENT/updated value -> the world changed; retire
                the stale fact(s) with a provenance trail, keep the new one.
                (This is the pin that disappears and gets replaced on screen.)
  - merge     : same aspect, SAME value -> duplicate/confirmation; fold into the
                existing record and bump corroboration (no second copy).
  - coexist   : no conflict -> keep both.

Two-stage, cheap-first:
  1. Candidate detection = vector similarity + same subject + same fact_type.
     No model call. (Cheap, and narrows a big store to a handful.)
  2. Relationship classification = ONE complete_json call on the main model,
     returning a decision + a one-line rationale (auditable, judge-explainable).

The model judges SEMANTICS only. Whether a supersede is actually applied is a
CODE decision with guards:
  - the new fact must be strictly fresher (more recent observed_at), and
  - a well-corroborated stored fact resists being overturned by a single
    low-confidence source.
On any ambiguity or parse failure we default to COEXIST — we never destroy
memory on uncertainty.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from momento.models.base import LLMBackend
from momento.memory.schema import MemoryRecord, Tier
from momento.memory import scoring
from momento.memory.store import MemoryStore

# --- tuning knobs -----------------------------------------------------
MAX_CANDIDATES = 5            # how many similar facts to weigh per ingest
CORROB_RESIST = 3            # a fact confirmed by >= this many sources resists...
CONF_MIN_TO_OVERTURN = 0.6   # ...being overturned by a source weaker than this
SIM_MAX_DISTANCE = 1.4       # ignore candidates farther than this (loose gate;
                             #   the model makes the real call)


class Decision(str, Enum):
    """What the model proposes."""
    SUPERSEDE = "supersede"
    MERGE = "merge"
    COEXIST = "coexist"


class Outcome(str, Enum):
    """What actually happened after the guards."""
    NO_CONFLICT = "no_conflict"   # nothing similar was stored
    COEXIST = "coexist"           # kept both, no conflict
    MERGED = "merged"             # folded into an existing record
    SUPERSEDED = "superseded"     # old fact(s) retired, new one wins
    CONTESTED = "contested"       # conflict found but a guard blocked retirement


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ReconciliationEvent:
    """The audit artifact for one ingest. Returned to the caller and appended
    to Reconciler.log; the demo reads .caption() for the on-screen message."""
    new_text: str
    new_id: str
    decision: Decision | None
    outcome: Outcome
    rationale: str = ""
    retired: list[tuple[str, str]] = field(default_factory=list)      # (id, text)
    merged_into: tuple[str, str] | None = None                        # (id, text)
    contested: list[tuple[str, str]] = field(default_factory=list)    # (id, text)
    guard_notes: list[str] = field(default_factory=list)
    at: datetime = field(default_factory=_now)

    def caption(self) -> str:
        if self.outcome is Outcome.SUPERSEDED:
            old = self.retired[0][1] if self.retired else "a stored fact"
            return f"Updated — retired: “{old}” → now: “{self.new_text}”. {self.rationale}"
        if self.outcome is Outcome.CONTESTED:
            note = f" ({self.guard_notes[0]})" if self.guard_notes else ""
            return f"Conflicting report noted but NOT applied: {self.rationale}{note}"
        if self.outcome is Outcome.MERGED:
            return f"Confirmed existing fact (corroboration bumped): “{self.new_text}”."
        if self.outcome is Outcome.COEXIST:
            return f"Added (no conflict): “{self.new_text}”."
        return f"Learned: “{self.new_text}”."


def _build_prompt(new: MemoryRecord, candidates: list[MemoryRecord]) -> list[dict]:
    listing = "\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(candidates))
    system = (
        "You reconcile a NEWLY OBSERVED travel fact against EXISTING stored facts "
        "about the same subject and category. Judge only the SEMANTIC relationship "
        "of the new fact to the existing ones — do NOT consider which is newer or "
        "more trustworthy.\n\n"
        "Choose exactly one decision:\n"
        "- \"supersede\": the new fact concerns the SAME aspect as one or more "
        "existing facts but states a DIFFERENT or UPDATED value (the situation "
        "changed; those facts are now outdated). List their indices.\n"
        "- \"merge\": the new fact concerns the SAME aspect and the SAME value as an "
        "existing fact (a duplicate or reworded/more-detailed confirmation). List "
        "the single index.\n"
        "- \"coexist\": the new fact does not conflict with any existing fact "
        "(different aspect/place/thing, or all independently true). Empty list.\n\n"
        "Respond with ONLY JSON, no prose:\n"
        '{"decision": "supersede|merge|coexist", "targets": [<indices>], '
        '"rationale": "<one short sentence>"}'
    )
    user = (
        f"Existing stored facts about this subject:\n{listing}\n\n"
        f"New observed fact:\n{new.text}"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


class Reconciler:
    def __init__(self, store: MemoryStore, backend: LLMBackend):
        self.store = store
        self.backend = backend
        self.log: list[ReconciliationEvent] = []

    # --- stage 1: cheap candidate detection ---------------------------
    def _candidates(self, new: MemoryRecord, user_id: str) -> list[MemoryRecord]:
        hits = self.store.search(
            new.text, k=MAX_CANDIDATES, user_id=user_id,
            subject=new.subject, fact_type=new.fact_type,
        )
        # enforce strict subject match (incl. None == None) and distance gate
        return [r for r, dist in hits
                if r.id != new.id and r.subject == new.subject
                and dist <= SIM_MAX_DISTANCE]

    # --- stage 2: model classification --------------------------------
    def _classify(self, new: MemoryRecord,
                  candidates: list[MemoryRecord]) -> tuple[Decision, list[MemoryRecord], str]:
        try:
            data = self.backend.complete_json(
                _build_prompt(new, candidates), fast=False, temperature=0.0)
        except Exception:
            return Decision.COEXIST, [], "classifier unavailable; defaulted to coexist"

        raw = str(data.get("decision", "")).strip().lower()
        try:
            decision = Decision(raw)
        except ValueError:
            decision = Decision.COEXIST
        rationale = str(data.get("rationale", "")).strip()

        targets: list[MemoryRecord] = []
        for idx in data.get("targets", []) or []:
            if isinstance(idx, int) and 1 <= idx <= len(candidates):
                targets.append(candidates[idx - 1])
        return decision, targets, rationale

    # --- guard: should the new fact actually overturn this one? --------
    @staticmethod
    def _new_should_win(new: MemoryRecord, target: MemoryRecord) -> tuple[bool, bool, bool]:
        fresher = new.observed_at > target.observed_at
        resisted = (target.corroboration >= CORROB_RESIST
                    and new.confidence < CONF_MIN_TO_OVERTURN)
        return (fresher and not resisted), fresher, resisted

    # --- main ----------------------------------------------------------
    def reconcile(self, new: MemoryRecord, *, user_id: str = "default") -> ReconciliationEvent:
        candidates = self._candidates(new, user_id)

        # No similar stored fact -> just learn it.
        if not candidates:
            self.store.add(new)
            ev = ReconciliationEvent(new.text, new.id, None, Outcome.NO_CONFLICT)
            self.log.append(ev); return ev

        decision, targets, rationale = self._classify(new, candidates)

        # --- MERGE: fold into the existing record, don't duplicate ---
        if decision is Decision.MERGE and targets:
            survivor = targets[0]
            if new.observed_at > survivor.observed_at:
                survivor.text = new.text
                survivor.observed_at = new.observed_at
            survivor.corroboration += 1
            survivor.confidence = max(survivor.confidence, new.confidence)
            scoring.rescore(survivor)
            self.store.update(survivor)
            ev = ReconciliationEvent(new.text, new.id, decision, Outcome.MERGED,
                                     rationale, merged_into=(survivor.id, survivor.text))
            self.log.append(ev); return ev

        # --- SUPERSEDE: retire stale fact(s), guarded ---
        if decision is Decision.SUPERSEDE and targets:
            retired, contested, notes = [], [], []
            for t in targets:
                win, fresher, resisted = self._new_should_win(new, t)
                t.contradictions += 1
                if win:
                    t.superseded_by = new.id
                    scoring.rescore(t)
                    t.tier = Tier.COLD                 # retired
                    self.store.update(t)
                    new.supersedes.append(t.id)
                    retired.append((t.id, t.text))
                else:
                    scoring.rescore(t)                 # contradiction lowers its score
                    self.store.update(t)
                    contested.append((t.id, t.text))
                    if resisted:
                        notes.append(f"kept a fact corroborated ×{t.corroboration}; new "
                                     f"source confidence {new.confidence:.2f} too low to overturn")
                    elif not fresher:
                        notes.append("new fact is not more recent than the stored one")
            self.store.add(new)                        # the new fact is now knowledge either way
            outcome = Outcome.SUPERSEDED if retired else Outcome.CONTESTED
            ev = ReconciliationEvent(new.text, new.id, decision, outcome, rationale,
                                     retired=retired, contested=contested, guard_notes=notes)
            self.log.append(ev); return ev

        # --- COEXIST (or empty-target supersede/merge) ---
        self.store.add(new)
        ev = ReconciliationEvent(new.text, new.id, decision, Outcome.COEXIST, rationale)
        self.log.append(ev); return ev


def supersede_trail(store: MemoryStore, *, user_id: str = "default",
                    subject: str | None = None) -> list[dict]:
    """Reconstruct the durable audit trail from the store: every retired fact,
    what replaced it, and the sources on each side. Powers the auditability
    story and the demo's 'retired because…' history."""
    recs = store.all(include_inactive=True)
    by_id = {r.id: r for r in recs}
    trail = []
    for r in recs:
        if r.user_id != user_id or not r.superseded_by:
            continue
        if subject is not None and r.subject != subject:
            continue
        nxt = by_id.get(r.superseded_by)
        trail.append({
            "retired": r.text,
            "replaced_by": nxt.text if nxt else None,
            "fact_type": r.fact_type.value,
            "subject": r.subject,
            "source_of_retired": r.provenance.source if r.provenance else None,
            "source_of_new": nxt.provenance.source if nxt and nxt.provenance else None,
        })
    return trail