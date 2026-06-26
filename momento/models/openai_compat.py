"""One backend for BOTH local Ollama and Qwen Cloud.

Both expose an OpenAI-compatible API, so the only differences are base URL,
API key, and model names — all from config. To add a provider that does NOT
speak the OpenAI protocol, write a new LLMBackend subclass instead."""
from openai import OpenAI
from momento.models.base import LLMBackend


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, *, base_url: str, api_key: str,
                 chat_model: str, fast_model: str, embed_model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.chat_model = chat_model    # heavier model for reasoning
        self.fast_model = fast_model    # cheap model for classification
        self.embed_model = embed_model

    def chat(self, messages: list[dict], *, temperature: float = 0.7,
             max_tokens: int = 1024, fast: bool = False) -> str:
        resp = self.client.chat.completions.create(
            model=self.fast_model if fast else self.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]