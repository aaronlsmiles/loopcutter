"""Sample-accurate loop extraction from a manifest."""

__version__ = "0.1.0"

from .cut import cut_loop
from .manifest import LoopSpec, load_manifest
from .keys import Key, parse_key, plan_sessions
from .verify import verify_cut

__all__ = [
    "cut_loop", "load_manifest", "LoopSpec", "verify_cut",
    "Key", "parse_key", "plan_sessions",
]
