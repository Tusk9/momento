"""Model-layer interface. Every backend implements this, so the rest of the
agent never knows whether it's talking to local Ollama or Qwen Cloud."""
from abc import ABC, abstractmethod
import json


class LLMBackend(ABC):
    """The one interface the whole agent depends on.

    Any model provider must give us two things:
      - chat():  free-form text (agent reasoning)
      - embed(): text -> vectors (memory store + retrieval)
    complete_json() is built on chat() and shared by all backends.
    """

    @abstractmethod
    def chat(self, messages: list[dict], *, temperature: float = 0.7,
             max_tokens: int = 1024, fast: bool = False) -> str:
        """Text reply to a list of {role, content} messages.
        fast=True uses the cheap model (for intent classification)."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """One embedding vector per input string."""
        ...

    def complete_json(self, messages: list[dict], *, temperature: float = 0.0,
                      fast: bool = False) -> dict:
        """Ask for JSON and parse it. Used by the intent classifier and the
        contradiction reconciler, which need structured output."""
        raw = self.chat(messages, temperature=temperature, fast=fast)
        return _parse_json(raw)


def _parse_json(raw: str) -> dict:
    """Models sometimes wrap JSON in ```json fences or add stray prose.
    Strip fences, then fall back to grabbing the outermost braces."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[len("json"):]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise