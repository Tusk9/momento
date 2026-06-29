"""Single place that decides which model backend the agent uses.
Flip MODEL_BACKEND in .env between 'ollama' (dev) and 'qwen_cloud' (judged
build) — nothing else in the codebase changes."""
import os
from functools import lru_cache
from dotenv import load_dotenv

from momento.models.openai_compat import OpenAICompatibleBackend
from momento.memory.schema import FactType

load_dotenv()

BACKEND = os.getenv("MODEL_BACKEND", "ollama").lower()

# --- Local dev: Ollama (OpenAI-compatible server on :11434) ---
OLLAMA = dict(
    base_url="http://localhost:11434/v1",
    api_key="ollama",                 # ignored by Ollama, but the SDK requires a value
    chat_model="qwen3",
    fast_model="qwen3",
    embed_model="nomic-embed-text",
)

# --- Judged build: Qwen Cloud / Alibaba Model Studio ---
# The four values below are PLACEHOLDERS — confirm exact IDs + base_url (region)
# on Model Studio when we wire prod.
QWEN_CLOUD = dict(
    base_url=os.getenv("QWEN_BASE_URL",
                       "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("QWEN_API_KEY", ""),
    chat_model="qwen3.7-plus",        # reasoning
    fast_model="qwen-flash",          # cheap classification
    embed_model="text-embedding-v4",  # VERIFY current ID + dim
)

# --- Memory: per-fact-type freshness half-lives (days) ---
# The "forgets on purpose" knob. STABLE facts decay slowly or never; VOLATILE
# facts decay fast. Used by memory/scoring.py. Tune freely.
HALF_LIVES_DAYS: dict[FactType, float] = {
    FactType.PREFERENCE:   3650,    # ~never (a stated taste persists)
    FactType.USER_PROFILE: 3650,
    FactType.VISITED:      36500,   # never
    FactType.GEOGRAPHIC:   36500,   # never
    FactType.POI_FACT:     365,     # slow
    FactType.VISA_RULE:    30,      # entry rules — weeks
    FactType.HOURS:        7,       # opening hours — days
    FactType.PRICE:        7,
    FactType.CLOSURE:      21,
    FactType.TRANSIT:      21,
    FactType.ADVISORY:     5,       # safety/weather — days
}

# Embedding dimension MUST match the active embed model (baked into the vector
# table at creation). 768 = nomic-embed-text (local, verified). For Qwen
# text-embedding-v4, VERIFY the dim on Model Studio; override via EMBED_DIM env.
EMBED_DIM = int(os.getenv("EMBED_DIM", "768" if BACKEND == "ollama" else "1024"))

# Local memory database file (gitignored via data/*.db).
DB_PATH = os.getenv("MOMENTO_DB", "data/memory.db")


@lru_cache
def get_backend() -> OpenAICompatibleBackend:
    cfg = OLLAMA if BACKEND == "ollama" else QWEN_CLOUD
    return OpenAICompatibleBackend(**cfg)