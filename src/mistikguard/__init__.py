"""Mistikguard — trustworthy memory & fabrication detection for LLM companions.

Governs what an LLM-backed assistant may treat as durable memory: provenance
(confirmed vs inferred), a deterministic write-gate, correction tombstones,
and an output-side grounding audit. The model proposes; Mistikguard decides.
"""
from . import memory_audit
from . import audit_judge
from . import memory_gate
from . import long_memory
from . import corrections
from .storage import safe_load_json, safe_save_json

__version__ = "0.1.0"
