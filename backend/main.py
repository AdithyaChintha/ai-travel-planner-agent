"""
FastAPI backend - ties together the graph, session memory, persisted
storage, and scheduler.

Endpoints:
    POST /plan   - run the trip-planning graph, persist + schedule the trip
    POST /ask    - answer a follow-up question about a trip
    GET  /trips  - list active trips (for the frontend to pick from)

Session registry:
    `sessions` is an in-memory dict {trip_id: {"memory": SessionMemory,
    "token_tracker": TokenTracker}}. It lets /ask use FAISS-based retrieval
    (memory.recall) for trips planned in this server process. If the
    process restarted, /ask falls back to the search_results stored in
    data/trips.json instead.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.config import settings
from backend.graph import build_graph
from backend.llm import get_llm
from backend.scheduler import register_trip_job, scheduler, start_scheduler
from backend.state import TripState
from backend.storage import get_trip, list_active_trips, save_trip
from backend.token_tracker import TokenTracker, track_usage
from backend.vector_store import SessionMemory

graph_app = build_graph()

# trip_id -> {"memory": SessionMemory, "token_tracker": TokenTracker}
sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    scheduler.shutdown()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


class PlanRequest(BaseModel):
    origin: str
    destination: str
    start_date: str  # "YYYY-MM-DD"
    end_date: str    # "YYYY-MM-DD"
    budget: float
    email: str
    preferences: str = ""


class PlanResponse(BaseModel):
    trip_id: str
    itinerary: str
    estimated_cost: float
    over_budget: bool
    budget_notes: str
    retry_count: int
    token_usage: dict


class AskRequest(BaseModel):
    trip_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    token_usage: dict


@app.post("/plan", response_model=PlanResponse)
def plan_trip(req: PlanRequest) -> PlanResponse:
    memory = SessionMemory()
    token_tracker = TokenTracker()

    initial_state: TripState = {
        "origin": req.origin,
        "destination": req.destination,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "budget": req.budget,
        "email": req.email,
        "preferences": req.preferences,
        "search_results": [],
        "itinerary": "",
        "estimated_cost": 0.0,
        "over_budget": False,
        "budget_notes": "",
        "retry_count": 0,
        "retry": False,
        "memory": memory,
        "token_tracker": token_tracker,
    }

    final_state = graph_app.invoke(initial_state)

    trip_id = save_trip({
        "origin": final_state["origin"],
        "destination": final_state["destination"],
        "start_date": final_state["start_date"],
        "end_date": final_state["end_date"],
        "budget": final_state["budget"],
        "email": final_state["email"],
        "preferences": final_state["preferences"],
        "itinerary": final_state["itinerary"],
        "estimated_cost": final_state["estimated_cost"],
        "search_results": final_state["search_results"],
    })

    sessions[trip_id] = {"memory": memory, "token_tracker": token_tracker}

    register_trip_job({
        "trip_id": trip_id,
        "destination": req.destination,
        "start_date": req.start_date,
        "email": req.email,
    })

    return PlanResponse(
        trip_id=trip_id,
        itinerary=final_state["itinerary"],
        estimated_cost=final_state["estimated_cost"],
        over_budget=final_state["over_budget"],
        budget_notes=final_state["budget_notes"],
        retry_count=final_state["retry_count"],
        token_usage=token_tracker.summary(),
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest) -> AskResponse:
    trip = get_trip(req.trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")

    session = sessions.get(req.trip_id)
    if session is not None:
        # Same server process as /plan - use FAISS retrieval.
        context_chunks = [doc.page_content for doc, _ in session["memory"].recall(req.question, k=3)]
        token_tracker = session["token_tracker"]
    else:
        # Server restarted since /plan - fall back to trips.json.
        context_chunks = [r["content"][:300] for r in trip.get("search_results", [])[:3]]
        token_tracker = TokenTracker()

    context = "\n\n".join(f"- {chunk}" for chunk in context_chunks)

    prompt = (
        f"You are helping with a trip to {trip['destination']} "
        f"({trip['start_date']} to {trip['end_date']}).\n\n"
        f"Itinerary:\n{trip['itinerary']}\n\n"
        f"Additional research:\n{context}\n\n"
        f"Question: {req.question}\n\n"
        "Answer concisely based on the above. If the answer isn't covered "
        "by it, say so and give your best general advice."
    )

    llm = get_llm()
    response = llm.invoke(prompt)
    track_usage(response.usage_metadata, token_tracker)

    return AskResponse(answer=response.content, token_usage=token_tracker.summary())


@app.get("/trips")
def trips() -> list[dict]:
    return list_active_trips()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
