"""Inspect the prediction map. From repo root: python test_transitions.py"""
from momento.intent.taxonomy import Intent
from momento.intent.transitions import predict_next

for current in [Intent.DESTINATION_DISCOVERY, Intent.ACCOMMODATION,
                Intent.ENTRY_REQUIREMENTS, Intent.PREFERENCE]:
    print(f"\nAfter '{current.value}', likely next:")
    for intent, p in predict_next(current, top_n=4):
        print(f"  {p:5.1%}  {intent.value}")