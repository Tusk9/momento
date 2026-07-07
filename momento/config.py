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
    api_key="ollama",
    chat_model="qwen3",
    fast_model="qwen3",
    embed_model="nomic-embed-text",
    embed_dim=768,
    send_dimensions=False,   # Ollama doesn't accept the dimensions param
)

# --- Judged build: Qwen Cloud (verified against Qwen Cloud docs) ---
QWEN_CLOUD = dict(
    base_url=os.getenv("QWEN_BASE_URL",
                       "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("QWEN_API_KEY", ""),
    chat_model="qwen3.7-plus",
    fast_model="qwen-flash",
    embed_model="text-embedding-v4",
    embed_dim=768,            # match local nomic-embed-text -> same DB works
    send_dimensions=True,     # v4 defaults to 1024; request 768 explicitly
)

# Embedding dimension for the vector table (must match the backend above).


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
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Local memory database file (gitignored via data/*.db).
DB_PATH = os.getenv("MOMENTO_DB", "data/memory.db")


@lru_cache
def get_backend() -> OpenAICompatibleBackend:
    cfg = OLLAMA if BACKEND == "ollama" else QWEN_CLOUD
    return OpenAICompatibleBackend(**cfg)