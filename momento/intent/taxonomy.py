"""The vocabulary the agent uses to understand each user turn.

Every turn is reduced to three things by the classifier:
  - intent:   what the user is trying to do right now (drives routing + prefetch)
  - entities: the concrete things mentioned (places, dates, cuisines, ...)
  - horizon:  how time-sensitive the turn is (drives how hard we check freshness)

Keep these names stable — the transition map, the prefetcher, and the memory
scorer all key off these exact strings.
"""
from enum import Enum


class Intent(str, Enum):
    # --- shaping the trip ---
    DESTINATION_DISCOVERY = "destination_discovery"  # where should I go / comparing places
    TRIP_FRAMING = "trip_framing"                    # dates, length, budget, who's coming, purpose
    # --- filling the trip ---
    ACCOMMODATION = "accommodation"                  # where to stay, neighborhoods, hotels
    ATTRACTIONS = "attractions"                      # sights, activities, tours, experiences
    DINING = "dining"                                # restaurants, food, where/what to eat
    ITINERARY = "itinerary"                          # sequencing days, routing, plotting a map
    TRANSPORT = "transport"                          # getting there and getting around
    # --- volatile real-world facts (freshness matters most here) ---
    ENTRY_REQUIREMENTS = "entry_requirements"        # visa, passport, customs, entry rules
    SAFETY_HEALTH = "safety_health"                  # advisories, weather warnings, vaccines
    PRACTICALITIES = "practicalities"                # money, SIM, plugs, etiquette, packing, current hours
    # --- about the user / the conversation itself ---
    PREFERENCE = "preference"                        # a durable like/dislike/constraint the user states
    RECAP = "recap"                                  # revisiting a past trip or earlier plan
    OTHER = "other"                                  # greetings, off-topic, fits nothing above


INTENT_DESCRIPTIONS: dict[Intent, str] = {
    Intent.DESTINATION_DISCOVERY: "Choosing or comparing where to travel.",
    Intent.TRIP_FRAMING: "Setting trip basics: dates, length, budget, companions, purpose.",
    Intent.ACCOMMODATION: "Where to stay — hotels, areas, lodging.",
    Intent.ATTRACTIONS: "Things to see and do — sights, activities, tours.",
    Intent.DINING: "Food and restaurants.",
    Intent.ITINERARY: "Ordering days, routing between places, building a map or schedule.",
    Intent.TRANSPORT: "Flights, trains, transit, car — getting there and around.",
    Intent.ENTRY_REQUIREMENTS: "Visas, passports, customs, border/entry rules.",
    Intent.SAFETY_HEALTH: "Safety advisories, weather warnings, health and vaccines.",
    Intent.PRACTICALITIES: "Money, SIM/data, plugs, language, etiquette, packing, current opening hours.",
    Intent.PREFERENCE: "The user states a lasting preference or constraint (e.g. 'I'm vegetarian', 'I hate early flights').",
    Intent.RECAP: "Returning to a previous trip or an earlier part of the plan.",
    Intent.OTHER: "Greetings, small talk, or anything that fits none of the above.",
}


class Horizon(str, Enum):
    NOW = "now"                  # about the present moment, or a trip in progress
    TRIP = "trip"                # about a specific, dated/committed upcoming trip
    EXPLORATORY = "exploratory"  # dreaming or comparing, no committed trip yet
    GENERAL = "general"          # not tied to a date (a preference, a geographic fact)


HORIZON_DESCRIPTIONS: dict[Horizon, str] = {
    Horizon.NOW: "Right now, or a trip currently underway (today's weather, is it open now).",
    Horizon.TRIP: "A specific upcoming trip with rough or fixed dates.",
    Horizon.EXPLORATORY: "Just exploring or comparing ideas, no dates committed.",
    Horizon.GENERAL: "Timeless — a durable preference or a fact that doesn't expire.",
}

# Entity types the classifier pulls out of a turn.
ENTITY_TYPES: dict[str, str] = {
    "place": "A city, region, country, or neighborhood.",
    "poi": "A specific named place (museum, restaurant, landmark, hotel).",
    "timeframe": "A date, month, season, or trip length.",
    "budget": "A spending level or amount.",
    "companion": "Who is travelling (partner, kids, friends, solo).",
    "cuisine": "A food type or dietary need (vegetarian, sushi, halal).",
    "activity": "An activity or interest (hiking, museums, nightlife).",
    "transport": "A mode of travel (flight, train, rental car, metro).",
}