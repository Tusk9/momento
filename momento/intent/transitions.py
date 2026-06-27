"""Loop-1 prediction signal: given the current intent, what is the user
likely to ask NEXT? The prefetcher uses this to warm memory before reasoning.

Weights are a HAND-CODED prior (deliberate: a learned model needs session
data we won't have early). `predict_next` is the stable seam — a learned
transition model can replace the lookup later without touching callers.
"""
from momento.intent.taxonomy import Intent

# How much of the prediction comes from the learned/prior map vs. a flat
# base rate that keeps every intent slightly reachable (users can jump).
_PRIOR_WEIGHT = 0.85
_BASE_RATE_WEIGHT = 0.15

# current_intent -> {likely_next_intent: weight}. Weights per row need not
# sum to 1; they're normalized in predict_next. Omitted targets get 0 prior
# (but still receive base rate). Read each row as "after X, users often want…"
_TRANSITIONS: dict[Intent, dict[Intent, float]] = {
    Intent.DESTINATION_DISCOVERY: {
        Intent.TRIP_FRAMING: 0.40, Intent.ATTRACTIONS: 0.25,
        Intent.ENTRY_REQUIREMENTS: 0.15, Intent.ACCOMMODATION: 0.10,
        Intent.DINING: 0.10,
    },
    Intent.TRIP_FRAMING: {
        Intent.ACCOMMODATION: 0.30, Intent.ATTRACTIONS: 0.25,
        Intent.TRANSPORT: 0.20, Intent.ENTRY_REQUIREMENTS: 0.15,
        Intent.ITINERARY: 0.10,
    },
    Intent.ACCOMMODATION: {
        Intent.ATTRACTIONS: 0.35, Intent.DINING: 0.25,
        Intent.ITINERARY: 0.20, Intent.TRANSPORT: 0.20,
    },
    Intent.ATTRACTIONS: {
        Intent.ITINERARY: 0.35, Intent.DINING: 0.30,
        Intent.TRANSPORT: 0.20, Intent.ACCOMMODATION: 0.15,
    },
    Intent.DINING: {
        Intent.ATTRACTIONS: 0.35, Intent.ITINERARY: 0.30,
        Intent.PRACTICALITIES: 0.20, Intent.TRANSPORT: 0.15,
    },
    Intent.ITINERARY: {
        Intent.TRANSPORT: 0.30, Intent.DINING: 0.25,
        Intent.ATTRACTIONS: 0.25, Intent.PRACTICALITIES: 0.20,
    },
    Intent.TRANSPORT: {
        Intent.ITINERARY: 0.30, Intent.PRACTICALITIES: 0.25,
        Intent.ACCOMMODATION: 0.25, Intent.ATTRACTIONS: 0.20,
    },
    Intent.ENTRY_REQUIREMENTS: {
        Intent.SAFETY_HEALTH: 0.30, Intent.PRACTICALITIES: 0.30,
        Intent.TRANSPORT: 0.20, Intent.TRIP_FRAMING: 0.20,
    },
    Intent.SAFETY_HEALTH: {
        Intent.PRACTICALITIES: 0.35, Intent.ENTRY_REQUIREMENTS: 0.25,
        Intent.ITINERARY: 0.20, Intent.TRIP_FRAMING: 0.20,
    },
    Intent.PRACTICALITIES: {
        Intent.ITINERARY: 0.30, Intent.ATTRACTIONS: 0.25,
        Intent.TRANSPORT: 0.25, Intent.DINING: 0.20,
    },
    # A stated preference rarely predicts the next topic — it reshapes ALL of
    # them. So lean mostly on the base rate, nudging toward planning intents.
    Intent.PREFERENCE: {
        Intent.ATTRACTIONS: 0.30, Intent.DINING: 0.30,
        Intent.ITINERARY: 0.20, Intent.ACCOMMODATION: 0.20,
    },
    # Recap and Other carry little predictive signal -> empty prior, so they
    # resolve almost entirely to the base rate (every intent stays reachable).
    Intent.RECAP: {},
    Intent.OTHER: {},
}

_ALL_INTENTS = list(Intent)
_FLAT = 1.0 / len(_ALL_INTENTS)


def predict_next(current: Intent, *, top_n: int | None = None) -> list[tuple[Intent, float]]:
    """Ranked (intent, probability) for what the user likely asks next.
    Probabilities sum to 1. top_n trims to the highest scorers (what the
    prefetcher will actually warm)."""
    prior = _TRANSITIONS.get(current, {})
    prior_total = sum(prior.values()) or 1.0

    scored: list[tuple[Intent, float]] = []
    for intent in _ALL_INTENTS:
        p_prior = prior.get(intent, 0.0) / prior_total
        score = _PRIOR_WEIGHT * p_prior + _BASE_RATE_WEIGHT * _FLAT
        scored.append((intent, score))

    total = sum(s for _, s in scored) or 1.0
    scored = [(i, s / total) for i, s in scored]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n] if top_n else scored


def top_intents(current: Intent, k: int = 3) -> list[Intent]:
    """Just the intent labels for the k most likely next turns."""
    return [intent for intent, _ in predict_next(current, top_n=k)]