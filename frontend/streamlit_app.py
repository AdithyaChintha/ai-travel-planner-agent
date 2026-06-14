"""
Streamlit frontend.

Talks to the FastAPI backend (backend/main.py) over HTTP:
    POST /plan -> plan a trip, show itinerary + budget + token usage
    POST /ask  -> follow-up chat about the planned trip

Run the backend first (`python -m backend.main`), then in another terminal,
from the project root:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit only puts this file's own directory (frontend/) on sys.path -
# add the project root too, so `backend` is importable below.
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import streamlit as st

from backend.config import settings

API_URL = f"http://{settings.api_host}:{settings.api_port}"

st.set_page_config(page_title=settings.app_name, page_icon="\U0001f9f3")
st.title(settings.app_name)

# ---------------------------------------------------------------------------
# Plan a trip
# ---------------------------------------------------------------------------

with st.form("plan_form"):
    origin = st.text_input("Starting from", placeholder="e.g. Bangalore, India")
    destination = st.text_input("Destination", placeholder="e.g. Goa, India")

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date")
    end_date = col2.date_input("End date")

    budget = st.number_input("Budget (USD)", min_value=1.0, value=200.0, step=10.0)
    email = st.text_input("Email (for disruption alerts)")
    preferences = st.text_area("Preferences (optional)", placeholder="e.g. vegetarian, relaxed pace")

    submitted = st.form_submit_button("Plan my trip")

if submitted:
    if not origin or not destination or not email:
        st.error("Please fill in at least starting location, destination, and email.")
    elif end_date < start_date:
        st.error("End date must be on or after the start date.")
    else:
        with st.spinner("Researching and building your itinerary..."):
            response = requests.post(
                f"{API_URL}/plan",
                json={
                    "origin": origin,
                    "destination": destination,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "budget": budget,
                    "email": email,
                    "preferences": preferences,
                },
            )

        if response.status_code != 200:
            st.error(f"Planning failed: {response.text}")
        else:
            data = response.json()
            st.session_state.trip_id = data["trip_id"]
            st.session_state.itinerary = data["itinerary"]
            st.session_state.estimated_cost = data["estimated_cost"]
            st.session_state.over_budget = data["over_budget"]
            st.session_state.budget_notes = data["budget_notes"]
            st.session_state.retry_count = data["retry_count"]
            st.session_state.token_usage = data["token_usage"]
            st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Show results + follow-up chat
# ---------------------------------------------------------------------------

if "trip_id" in st.session_state:
    st.divider()
    st.subheader("Your Itinerary")
    st.markdown(st.session_state.itinerary)

    st.subheader("Budget")
    status = "Over budget" if st.session_state.over_budget else "Within budget"
    st.write(f"**Estimated cost:** ${st.session_state.estimated_cost:.2f} ({status})")
    st.caption(st.session_state.budget_notes)
    if st.session_state.retry_count > 0:
        st.caption(f"Replanned {st.session_state.retry_count} time(s) to try to fit the budget.")

    with st.expander("Token usage (this session)"):
        st.json(st.session_state.token_usage)

    st.divider()
    st.subheader("Ask a follow-up question")

    for question, answer in st.session_state.chat_history:
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Assistant:** {answer}")

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Your question", placeholder="e.g. What should I pack?")
        ask_submitted = st.form_submit_button("Ask")

    if ask_submitted and question:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"trip_id": st.session_state.trip_id, "question": question},
            )

        if response.status_code != 200:
            st.error(f"Question failed: {response.text}")
        else:
            data = response.json()
            st.session_state.chat_history.append((question, data["answer"]))
            st.session_state.token_usage = data["token_usage"]
            st.rerun()
