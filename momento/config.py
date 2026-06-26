"""Single place that decides which model backend the agent uses.
Flip MODEL_BACKEND in .env between 'ollama' (dev) and 'qwen_cloud' (judged
build) — nothing else in the codebase changes."""
import os
from functools import lru_cache
from dotenv import load_dotenv
from momento.models.openai_compat import OpenAICompatibleBackend

load_dotenv()

BACKEND = os.getenv("MODEL_BACKEND", "ollama").lower()

# --- Local dev: Ollama (OpenAI-compatible server on :11434) ---
OLLAMA = dict(
    base_url="http://localhost:11434/v1",
    api_key="ollama",                 # Ollama ignores it, but the SDK requires a value
    chat_model="qwen3",               # the tag you `ollama pull`ed
    fast_model="qwen3",               # same locally; a smaller tag is fine here
    embed_model="nomic-embed-text",
)

# --- Judged build: Qwen Cloud / Alibaba Model Studio ---
# All four values below are PLACEHOLDERS to confirm on Model Studio when we
# wire prod (model IDs, embedding model, and base_url differ by region).
QWEN_CLOUD = dict(
    base_url=os.getenv("QWEN_BASE_URL",
                       "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("QWEN_API_KEY", ""),
    chat_model="qwen3.7-plus",        # reasoning
    fast_model="qwen-flash",          # cheap intent classification
    embed_model="text-embedding-v4",  # VERIFY current ID on Model Studio
)


@lru_cache
def get_backend() -> OpenAICompatibleBackend:
    cfg = OLLAMA if BACKEND == "ollama" else QWEN_CLOUD
    return OpenAICompatibleBackend(**cfg)