import re
from ._log import dprint

# Phrases that signal the assistant is claiming a memory / shared past.
_CLAIM_PATTERNS = [
    # ── originals (unchanged) ──
    r"you (?:mentioned|told me|said|shared)",
    r"i remember (?:you|when|that|our|us|we)",
    r"i recall (?:you|when|that|our|us|we|how)",
    r"(?:last time|earlier|before|previously) you",
    r"you(?:'ve| have) (?:mentioned|told|said)",
    r"as you (?:mentioned|said|told)",
    r"we (?:talked|spoke|discussed|went|did|had|saw|met)",
    r"you and i (?:went|did|had|saw|met|talked|used)",
    r"you used to",
    r"i (?:know|remember) (?:that )?you(?:'re| are| have| had| went| did)",

    # ── NEW: entity-attribute memory claims ──
    r"i (?:remember|recall) \w+(?:'s)? (?:is|was|are|were|has|have|had|lives|lived|works|worked|likes|liked|plays|played|owns|owned)\b",

    # possessive/article-led memory claims
    r"i (?:remember|recall|know) (?:your|his|her|their)\b",
    r"you (?:have|had) (?:a|an|your)\b",

    # article-led memory claims ("I remember the trip", "I recall the marathon")
    r"i (?:remember|recall) the \w+",
]

def detect_memory_claims(reply: str):
    """Return list of (matched_phrase, surrounding_sentence) for memory-claims in the reply."""
    claims = []
    if not reply:
        return claims
    sentences = re.split(r'(?<=[.!?])\s+', reply)
    for sent in sentences:
        low = sent.lower()
        for pat in _CLAIM_PATTERNS:
            m = re.search(pat, low)
            if m:
                claims.append((m.group(0), sent.strip()))
                break
    return claims

def _normalize(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower())

def claim_is_grounded(claim_sentence: str, memory_texts: list, recent_user_msgs: list):
    """Heuristic: is this memory-claim supported by something in her actual memory
    or recent conversation? Returns True if plausibly grounded, False if likely fabricated.
    Conservative: only flags as ungrounded when it finds NO overlap at all."""
    if not claim_sentence or not isinstance(claim_sentence, str):
        return True  # nothing to flag
    memory_texts = memory_texts or []
    recent_user_msgs = recent_user_msgs or []
    claim_norm = _normalize(claim_sentence)
    stop = set("you i me my your we us the a an is are was were that this it to of and "
               "mentioned told said shared remember when last time earlier before previously "
               "have has about used talked spoke discussed as".split())
    claim_words = {w for w in claim_norm.split() if w not in stop and len(w) > 2}
    if not claim_words:
        dprint(f"[AUDIT] ✓ claim has no concrete words to ground — assuming grounded")
        return True  # nothing concrete claimed — can't flag
    haystack = " ".join(_normalize(t) for t in (memory_texts + recent_user_msgs))
    hay_words = set(haystack.split())
    overlap = claim_words & hay_words
    has_overlap = len(overlap) > 0
    if has_overlap:
        dprint(f"[AUDIT] ✓ claim GROUNDED: found {len(overlap)} matching words: {list(overlap)[:5]}")
    else:
        dprint(f"[AUDIT] ⚠️ claim UNGROUNDED: no overlapping words. Claim words: {claim_words}")
    return has_overlap

def audit_reply(reply: str, memory_texts: list, recent_user_msgs: list):
    """Returns list of ungrounded claims: [{'phrase':..., 'sentence':...}].
    Empty list = no fabricated memory-claims detected."""
    dprint(f"\n[AUDIT] ━━━━ AUDITING REPLY for memory-claims")
    ungrounded = []
    # Input validation: tolerate bad/None input gracefully.
    if not reply or not isinstance(reply, str):
        return ungrounded
    memory_texts = list(memory_texts) if memory_texts else []
    recent_user_msgs = list(recent_user_msgs) if recent_user_msgs else []
    claims = detect_memory_claims(reply)
    if not claims:
        dprint(f"[AUDIT] ✓ no memory-claims detected → reply passes tier-1 (free check)")
        return ungrounded

    dprint(f"[AUDIT] Found {len(claims)} memory-claim(s), running tier-1.5 word-overlap check...")
    for phrase, sentence in claims:
        dprint(f"[AUDIT]   Claim: '{phrase}' in '{sentence[:70]}...'")
        if not claim_is_grounded(sentence, memory_texts, recent_user_msgs):
            dprint(f"[AUDIT]   → FLAGGED for tier-2 LLM judge")
            ungrounded.append({"phrase": phrase, "sentence": sentence})
        else:
            dprint(f"[AUDIT]   → passes tier-1.5, accepted")

    if ungrounded:
        dprint(f"[AUDIT] ⚠️ {len(ungrounded)} claim(s) flagged for tier-2 (LLM judge)")
    else:
        dprint(f"[AUDIT] ✅ all claims grounded by tier-1.5")
    return ungrounded
