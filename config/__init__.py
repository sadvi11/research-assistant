from .env import load_env

# Make `cp .env.example .env` actually work: importing config loads it.
load_env()

from .settings import Settings, get_settings
from .sources import SourceTier, TrustPolicy, get_trust_policy

__all__ = ["Settings", "get_settings", "SourceTier", "TrustPolicy",
           "get_trust_policy", "load_env"]
