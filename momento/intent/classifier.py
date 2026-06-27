"""Turns a single user message into a structured Classification using the
CHEAP model (fast=True). This is the first real consumer of the model layer
and the entry point of the memory loop: its output decides what we prefetch
and how we score what we store.
"""
from dataclasses import dataclass, field

from momento.models.base import LLMBackend
from momento.intent.taxonomy import (
    Intent, Horizon, INTENT_DESCRIPTIONS, HORIZON_DESCRIPTIONS, ENTITY_TYPES,
)


@dataclass
class Classification:
    intent: Intent
    horizon: Horizon
    entities: list[dict] = field(default_factory=list)  # [{"type": ..., "value": ...}]
    raw: dict = field(default_factory=dict)             # model's unparsed output, for debugging


def _build_system_prompt() -> str:
    intents = "\n".join(f"  - {i.value}: {INTENT_DESCRIPTIONS[i]}" for i in Intent)
    horizons = "\n".join(f"  - {h.value}: {HORIZON_DESCRIPTIONS[h]}" for h in Horizon)
    entity_types = "\n".join(f"  - {name}: {desc}" for name, desc in ENTITY_TYPES.items())
    return f"""You label a single travel-planning message for a memory agent.

Pick exactly ONE intent:
{intents}

Pick exactly ONE horizon:
{horizons}

Extract entities mentioned in the message. Each entity is {{"type": <type>, "value": <text>}}.
Use only these types (skip anything that doesn't fit):
{entity_types}

Respond with ONLY a JSON object, no explanation and no reasoning, in exactly this shape:
{{"intent": "<intent>", "horizon": "<horizon>", "entities": [{{"type": "place", "value": "Kyoto"}}]}}
If there are no entities, use an empty list."""


SYSTEM_PROMPT = _build_system_prompt()


def classify(text: str, backend: LLMBackend,
             recent_intents: list[Intent] | None = None) -> Classification:
    """Classify one user turn. recent_intents (most-recent-last) is optional
    context that helps disambiguate short follow-ups like 'what about there?'."""
    user_block = text
    if recent_intents:
        trail = ", ".join(i.value for i in recent_intents[-3:])
        user_block = f"[recent intents: {trail}]\nMessage: {text}"

    data = backend.complete_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ],
        fast=True,
        temperature=0.0,
    )
    return _coerce(data)


def _coerce(data: dict) -> Classification:
    """Small models occasionally return a label that's slightly off. Validate
    against the enums and fall back safely rather than crash."""
    intent = _to_enum(Intent, data.get("intent"), default=Intent.OTHER)
    horizon = _to_enum(Horizon, data.get("horizon"), default=Horizon.GENERAL)

    entities = []
    for e in data.get("entities", []) or []:
        if isinstance(e, dict) and e.get("type") in ENTITY_TYPES and e.get("value"):
            entities.append({"type": e["type"], "value": str(e["value"])})

    return Classification(intent=intent, horizon=horizon, entities=entities, raw=data)


def _to_enum(enum_cls, value, *, default):
    if value is None:
        return default
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        return default