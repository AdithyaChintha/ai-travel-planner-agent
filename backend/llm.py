"""
Shared Gemini chat model client.

Every agent node imports get_llm() from here instead of constructing its
own ChatGoogleGenerativeAI - one place to configure the model, one place
to change it later.

LangChain wraps every provider behind the same interface:
    response = llm.invoke(...)
    response.content        -> the text reply
    response.usage_metadata  -> {"input_tokens":, "output_tokens":, "total_tokens":}
      (this is what token_tracker.py reads)
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings


def get_llm(temperature: float = 0.3, thinking_budget: int = 0) -> ChatGoogleGenerativeAI:
    """
    Returns a configured Gemini chat model.

    temperature: 0 = deterministic, 1 = more varied. Lower values suit
    factual itinerary/budget work better than creative writing.

    thinking_budget: max tokens Gemini 2.5 spends on internal "reasoning"
    before answering. Defaults to 0 (disabled) - without this, even a
    one-sentence reply costs 1000+ hidden "reasoning" tokens, which matters
    for free-tier quota and our token tracker. Pass a higher value for
    nodes that might benefit from extra reasoning (e.g. itinerary planning).
    """
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        thinking_budget=thinking_budget,
    )


if __name__ == "__main__":
    # Manual test: run `python llm.py`
    llm = get_llm()
    response = llm.invoke("In one sentence, what makes a good travel itinerary?")
    print("Response:", response.content)
    print("Token usage:", response.usage_metadata)
