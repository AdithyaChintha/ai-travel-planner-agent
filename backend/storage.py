"""
Persistent trip records: data/trips.json

This is the durable source of truth for every planned trip - it survives
server restarts. Used by:

- main.py: to save the result of /plan, and as a fallback for /ask when a
  trip's in-memory SessionMemory has been lost (server restarted, or the
  session simply aged out) - rebuilt from this file's search_results.
- scheduler.py: on startup, to re-register each active trip's recurring
  disruption-check job via list_active_trips().

Each trip record looks like:
{
  "trip_id": "a1b2c3d4",
  "origin": "Bangalore",
  "destination": "Goa",
  "start_date": "2026-07-01",
  "end_date": "2026-07-05",
  "budget": 500,
  "email": "you@example.com",
  "itinerary": "Day 1: ...",
  "estimated_cost": 480,
  "search_results": [ {"title": ..., "url": ..., "content": ...}, ... ],
  "created_at": "2026-06-14T10:30:00"
}
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TRIPS_FILE = DATA_DIR / "trips.json"


def _read_all() -> list[dict]:
    if not TRIPS_FILE.exists():
        return []
    with open(TRIPS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_all(trips: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(TRIPS_FILE, "w", encoding="utf-8") as f:
        json.dump(trips, f, indent=2)


def save_trip(trip: dict) -> str:
    """Save a new trip record. Adds trip_id and created_at, returns trip_id."""
    trip = dict(trip)
    trip["trip_id"] = uuid.uuid4().hex[:8]
    trip["created_at"] = datetime.now().isoformat()
    trip.setdefault("search_results", [])

    trips = _read_all()
    trips.append(trip)
    _write_all(trips)
    return trip["trip_id"]


def get_trip(trip_id: str) -> dict | None:
    """Look up a trip record by id, or None if it doesn't exist."""
    for trip in _read_all():
        if trip["trip_id"] == trip_id:
            return trip
    return None


def append_search_result(trip_id: str, result: dict) -> None:
    """Add a newly-discovered search result to a trip's search_results,
    so future follow-up questions can use it too."""
    trips = _read_all()
    for trip in trips:
        if trip["trip_id"] == trip_id:
            trip.setdefault("search_results", []).append(result)
            break
    _write_all(trips)


def list_active_trips() -> list[dict]:
    """Trips whose end_date hasn't passed yet - used by the scheduler on
    startup to re-register recurring disruption-check jobs."""
    today = date.today().isoformat()
    return [t for t in _read_all() if t["end_date"] >= today]


if __name__ == "__main__":
    # Manual test: run `python storage.py` - no API keys needed.
    trip_id = save_trip({
        "origin": "Bangalore",
        "destination": "Goa",
        "start_date": "2026-07-01",
        "end_date": "2026-07-05",
        "budget": 500,
        "email": "you@example.com",
        "itinerary": "Day 1: Arrive, check into hotel near Calangute...",
        "estimated_cost": 480,
        "search_results": [
            {"title": "Goa Budget Hotels Guide", "url": "https://example.com",
             "content": "Most guesthouses cost $15-25/night."},
        ],
    })
    print("Saved trip:", trip_id)
    print("Loaded back:", get_trip(trip_id))

    append_search_result(trip_id, {
        "title": "Goa Vegetarian Food Guide",
        "url": "https://example.com/veg",
        "content": "Most beach shacks offer a separate veg menu.",
    })
    print("After append, search_results:", get_trip(trip_id)["search_results"])

    print("Active trips:", [t["trip_id"] for t in list_active_trips()])
