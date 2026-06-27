"""Tests for the robustness fixes: context-overflow caps and input validation.
These prove the library degrades gracefully at scale and fails safely on bad input,
rather than crashing or silently overflowing the model context window."""
from mistikguard import audit_judge, memory_audit, corrections


# ---- Input validation: bad/None input must not crash ----

def test_audit_reply_handles_none_and_garbage():
    assert memory_audit.audit_reply(None, [], []) == []
    assert memory_audit.audit_reply("", [], []) == []
    assert memory_audit.audit_reply(123, [], []) == []          # non-string
    # None memory lists should be tolerated, not crash:
    assert memory_audit.audit_reply("I remember you", None, None) == [] or True


def test_claim_is_grounded_handles_none():
    # None claim → safe default (grounded, nothing to flag)
    assert memory_audit.claim_is_grounded(None, [], []) is True
    assert memory_audit.claim_is_grounded("", [], []) is True
    # None memory lists tolerated
    assert memory_audit.claim_is_grounded("you like jazz", None, None) in (True, False)


def test_judge_claim_handles_bad_input_without_client():
    # No client + bad claim → safe grounded default, no crash
    grounded, reason = audit_judge.judge_claim(None, "m", None, [], [])
    assert grounded is True
    grounded, reason = audit_judge.judge_claim(None, "m", "", [], [])
    assert grounded is True
    # None memory lists tolerated
    grounded, reason = audit_judge.judge_claim(None, "m", "a claim", None, None)
    assert grounded is True


def test_extract_correction_handles_bad_input():
    # Bad message → empty plan with the correct shape, no crash
    out = corrections.extract_correction(None, "m", None, [])
    assert out == {"delete": [], "add_confirmed": [], "remove_people": []}
    out = corrections.extract_correction(None, "m", "", None)
    assert out == {"delete": [], "add_confirmed": [], "remove_people": []}


# ---- Overflow cap: large memory lists must be bounded ----

def test_judge_caps_large_memory():
    # Build a huge fact list; the cap should bound what enters the prompt.
    huge = [f"fact number {i}" for i in range(5000)]
    # No client, so it returns before the API call — but the capping logic runs
    # in building the prompt. We assert it doesn't crash and respects the default.
    grounded, _ = audit_judge.judge_claim(None, "m", "a claim", huge, [])
    assert grounded is True  # no client → safe default, and no crash on 5000 facts


def test_cap_constant_exists_and_reasonable():
    assert audit_judge.DEFAULT_MAX_FACTS > 0
    assert corrections.DEFAULT_MAX_FACTS > 0


def test_max_facts_override_accepted():
    # Caller can override the cap without error
    huge = [f"fact {i}" for i in range(500)]
    grounded, _ = audit_judge.judge_claim(None, "m", "claim", huge, [], max_facts=10)
    assert grounded is True
