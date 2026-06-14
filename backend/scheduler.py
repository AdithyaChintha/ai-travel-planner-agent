"""
APScheduler: recurring disruption checks for upcoming trips.

For each trip, register_trip_job() schedules a recurring job:
    every settings.disruption_check_interval_days, starting "now" (the
    planning day) until trip["start_date"] (the destination day).

Each time the job fires:
    1. Tavily-search weather + travel-advisory news for the destination.
    2. Gemini (structured output) decides: is there a notable disruption?
    3. If yes, email trip["email"] with a summary. If no, do nothing -
       disruption checks are silent unless something is actually wrong.

APScheduler's jobs are in-memory only and don't survive a restart, so
main.py calls reregister_active_trips() on startup, which re-creates a job
for every trip in data/trips.json whose trip hasn't ended yet (via
storage.list_active_trips()).
"""

from __future__ import annotations

import smtplib
from datetime import datetime
from email.message import EmailMessage

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel, Field

from backend.config import settings
from backend.llm import get_llm
from backend.token_tracker import track_usage
from backend.tools import web_search

scheduler = BackgroundScheduler()


class DisruptionReport(BaseModel):
    disruption_found: bool = Field(
        description="True if there is a notable weather or geopolitical "
        "issue that could affect travel to the destination soon."
    )
    summary: str = Field(
        description="If disruption_found is True, a 1-2 sentence summary "
        "of the issue. Otherwise an empty string."
    )


def check_disruptions(destination: str) -> DisruptionReport:
    """Search for weather/travel-advisory news and ask Gemini whether
    there's a notable disruption for travelers heading to `destination`."""
    results: list[dict] = []
    for query in (f"{destination} weather forecast", f"{destination} travel advisory news"):
        results.extend(web_search(query))

    context = "\n\n".join(f"- {r['title']}: {r['content'][:300]}" for r in results)

    llm = get_llm()
    structured_llm = llm.with_structured_output(DisruptionReport, include_raw=True)
    prompt = (
        f"Here is recent weather and travel-advisory information about "
        f"{destination}:\n\n{context}\n\n"
        "Based on this, is there a notable weather or geopolitical "
        "disruption that could affect a traveler heading there soon?"
    )

    result = structured_llm.invoke(prompt)
    # No session is associated with this background job - only the global
    # tracker gets updated.
    track_usage(result["raw"].usage_metadata)

    return result["parsed"]


def send_email(to: str, subject: str, body: str) -> None:
    """Send a plain-text email via Gmail SMTP."""
    msg = EmailMessage()
    msg["From"] = settings.gmail_address
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    # Gmail App Passwords are often copy-pasted with spaces ("abcd efgh
    # ijkl mnop") - strip them so login doesn't fail on a formatting quirk.
    password = settings.gmail_app_password.replace(" ", "")

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(settings.gmail_address, password)
        smtp.send_message(msg)


def _run_disruption_check(destination: str, email: str) -> None:
    """The job function APScheduler runs on each interval for one trip."""
    report = check_disruptions(destination)
    if report.disruption_found:
        send_email(
            to=email,
            subject=f"Travel alert: possible disruption in {destination}",
            body=(
                f"Heads up - we found a possible disruption for your "
                f"upcoming trip to {destination}:\n\n{report.summary}"
            ),
        )


def register_trip_job(trip: dict) -> None:
    """Register (or re-register) a recurring disruption-check job for one
    trip. Does nothing if the trip's start date has already passed."""
    start_date = datetime.strptime(trip["start_date"], "%Y-%m-%d")
    if start_date <= datetime.now():
        return

    scheduler.add_job(
        _run_disruption_check,
        trigger=IntervalTrigger(
            days=settings.disruption_check_interval_days,
            start_date=datetime.now(),
            end_date=start_date,
        ),
        args=[trip["destination"], trip["email"]],
        id=f"disruption_check_{trip['trip_id']}",
        replace_existing=True,
    )


def reregister_active_trips() -> None:
    """Re-create jobs for every still-active trip in data/trips.json -
    called on app startup since APScheduler doesn't persist jobs."""
    from backend.storage import list_active_trips

    for trip in list_active_trips():
        register_trip_job(trip)


def start_scheduler() -> None:
    if not scheduler.running:
        scheduler.start()
    reregister_active_trips()


if __name__ == "__main__":
    # Manual test: run `python scheduler.py` (requires GOOGLE_API_KEY,
    # TAVILY_API_KEY, GMAIL_ADDRESS, GMAIL_APP_PASSWORD in .env).
    print("--- Checking for disruptions in Goa, India ---")
    report = check_disruptions("Goa, India")
    print(f"disruption_found: {report.disruption_found}")
    print(f"summary: {report.summary!r}")

    print("\n--- Sending a test email ---")
    send_email(
        to=settings.gmail_address,
        subject="Test email from AI Travel Planner scheduler",
        body="If you're reading this, send_email() works correctly.",
    )
    print(f"Sent test email to {settings.gmail_address}")
