"""Run sample turns through the classifier. From repo root: python test_classifier.py"""
from momento.config import get_backend
from momento.intent.classifier import classify

backend = get_backend()

samples = [
    "We're thinking about somewhere warm in Europe for a week in October.",
    "I'm vegetarian and I really don't like crowded tourist traps.",
    "Do US citizens need a visa to visit Japan?",
    "What are the best neighborhoods to stay in Kyoto?",
    "Can you map out a 3-day route hitting the temples we talked about?",
    "Is it safe to travel to Bali right now?",
]

for s in samples:
    c = classify(s, backend)
    ents = ", ".join(f'{e["type"]}={e["value"]}' for e in c.entities) or "—"
    print(f"\n> {s}")
    print(f"  intent={c.intent.value}  horizon={c.horizon.value}")
    print(f"  entities: {ents}")