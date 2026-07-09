"""
Centralized configuration for the AI Travel Planner.

All values are loaded from environment variables (or a local .env file).
Never hardcode API keys or passwords directly in code - they belong in .env,
which is excluded from git via .gitignore.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (one level up from this file, in backend/), so .env is found
# no matter what directory a command is run from.
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    # --- Gemini (LLM) ---
    google_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # --- Tavily (real-time web search) ---
    tavily_api_key: str
    # How many results to fetch per search. Kept small to stay
    # budget-friendly (free tier) and keep prompts to Gemini compact.
    tavily_max_results: int = 3

    # --- Gmail (for disruption alert emails) ---
    gmail_address: str
    gmail_app_password: str

    # --- FAISS (session memory for RAG) ---
    # Model used to turn text into vectors for similarity search.
    embedding_model: str = "models/gemini-embedding-001"
    # Directory where the FAISS index is persisted across sessions.
    faiss_store_path: Path = Path(__file__).parent.parent / "data" / "faiss_index"

    # --- Budget-check retry loop ---
    # Max number of times the graph will loop research -> itinerary ->
    # budget_check before giving up and returning the best plan found.
    budget_max_retries: int = 2

    # --- APScheduler ---
    # How often (in days) to check the destination for weather/geopolitical
    # disruptions, from the planning day until the trip's start date. An
    # email is only sent if a disruption is actually found.
    disruption_check_interval_days: int = 4

    # --- FastAPI ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- App ---
    app_name: str = "AI Travel Planner"

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


# A single shared instance, imported everywhere else in the project.
settings = Settings()
