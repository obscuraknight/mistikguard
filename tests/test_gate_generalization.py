"""Tests that the write-gate is genuinely generic — not tied to any one
assistant or user name. These exercise the generalization we did during
extraction (assistant_name / user_name / storage_path as config).
"""
import os
import tempfile
import pytest
from mistikguard import memory_gate as gate


def _cfg(assistant="Aria", user="Sarah"):
    """A gate config with custom names and a throwaway tombstone path."""
    tmp = tempfile.mkdtemp(prefix="mistikguard_test_")
    return gate.GateConfig(
        assistant_name=assistant,
        user_name=user,
        corrections_log_path=os.path.join(tmp, "corrections.json"),
    )


def test_self_narration_uses_configured_assistant_name():
    """Text starting with the assistant's name is rejected as self-narration —
    for a CUSTOM name, not just 'mistik'."""
    cfg = _cfg(assistant="Aria")
    allowed, reason = gate.gate_fact("Aria feels calm today.", [], config=cfg)
    assert not allowed, "should reject self-narration about the assistant"
    assert "noise" in reason


def test_other_assistant_name_not_falsely_flagged():
    """A real user fact that happens to mention a DIFFERENT name is allowed."""
    cfg = _cfg(assistant="Aria")
    allowed, _ = gate.gate_fact("User's brother is named Mistik.", [], config=cfg)
    assert allowed, "a fact about a person who happens to be named Mistik should pass"


def test_user_name_rejected_as_person():
    """The user themselves must never be stored as a 'person' in their life."""
    cfg = _cfg(user="Sarah")
    allowed, reason = gate.gate_person("Sarah", config=cfg)
    assert not allowed, "the user's own name should be rejected as a person"
    assert "non-name token" in reason


def test_real_person_allowed():
    """A genuine third party passes the people-gate."""
    cfg = _cfg(user="Sarah")
    allowed, _ = gate.gate_person("Olia", relationship="sister", config=cfg)
    assert allowed, "a real named person should be allowed"


def test_pronouns_still_rejected():
    """Generic non-name tokens are still rejected regardless of config."""
    cfg = _cfg()
    for junk in ["you", "they", "someone", "the"]:
        allowed, _ = gate.gate_person(junk, config=cfg)
        assert not allowed, f"'{junk}' should be rejected as a non-name"


def test_tombstone_blocks_reintroduction():
    """A tombstoned fact cannot be re-added — with a custom storage path."""
    cfg = _cfg()
    gate.record_correction_tombstone("user plays violin", config=cfg)
    allowed, reason = gate.gate_fact("user plays violin", [], config=cfg)
    assert not allowed, "tombstoned material must be rejected"
    assert "tombstone" in reason.lower() or "corrected" in reason.lower()


def test_contradiction_of_confirmed_fact():
    """A proposed fact that contradicts a confirmed one is rejected."""
    cfg = _cfg()
    confirmed = ["user lives in Berlin"]
    allowed, reason = gate.gate_fact("user lives in Lisbon", confirmed, config=cfg)
    assert not allowed, "should reject contradiction of a confirmed fact"
    assert "contradict" in reason.lower()


def test_default_config_still_works():
    """With no config passed, the module default applies and basic gating works."""
    allowed, _ = gate.gate_person("Olia", relationship="friend")
    assert allowed
    allowed, _ = gate.gate_person("you")
    assert not allowed
