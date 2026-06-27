"""LLM grounding-judge: given a memory-claim and her actual memory, decide if the
claim is SUPPORTED (grounded) or UNSUPPORTED (fabricated). More accurate than
word-overlap because it understands meaning, not just shared tokens."""
import json
from ._log import dprint

def judge_claim(client, model, claim_sentence, memory_texts, recent_user_msgs):
    """Returns (grounded: bool, reason: str).
    Conservative: defaults to grounded (True) on any error, so the judge never
    produces false fabrication-alarms when uncertain."""
    dprint(f"\n[JUDGE] ━━━━ TIER-2 LLM GROUNDING JUDGE")
    dprint(f"[JUDGE] Claim: '{claim_sentence[:80]}'")

    if client is None:
        dprint(f"[JUDGE] ⚠️ no client available → defaulting grounded (safe default)")
        return (True, "no client")
    
    facts_block = "\n".join(f"- {t}" for t in memory_texts if t) or "(no stored facts)"
    recent_block = "\n".join(f"- {m}" for m in recent_user_msgs[-6:] if m) or "(none)"
    prompt = (
        "You are a fact-checker for an AI companion's memory. The companion just wrote a "
        "sentence claiming to remember something about the user. Decide whether that claim "
        "is SUPPORTED by the companion's actual stored memory + recent conversation, or "
        "whether it is UNSUPPORTED (the companion is fabricating a memory that isn't there).\n\n"
        "Be fair: a claim is SUPPORTED if the stored facts or recent messages contain the "
        "information, even if worded differently. A claim is UNSUPPORTED only if there is NO "
        "basis for it in the memory below. Generic relationship statements with no specific "
        "factual content (e.g. 'we've talked before', 'I remember you') are always SUPPORTED — "
        "they assert nothing checkable.\n\n"
        f"STORED MEMORY:\n{facts_block}\n\n"
        f"RECENT USER MESSAGES:\n{recent_block}\n\n"
        f"THE CLAIM TO CHECK:\n\"{claim_sentence}\"\n\n"
        "Reply ONLY as JSON: {\"supported\": true/false, \"reason\": \"brief\"}"
    )
    last_err = None
    for attempt in range(2):  # original try + one retry
        try:
            if attempt > 0:
                dprint(f"[JUDGE] retry after parse/call failure...")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
            data = json.loads(raw)
            supported = bool(data.get("supported", True))
            reason = str(data.get("reason", ""))[:100]
            if supported:
                dprint(f"[JUDGE] ✅ SUPPORTED: {reason}")
            else:
                dprint(f"[JUDGE] ❌ UNSUPPORTED (FABRICATED): {reason}")
            return (supported, reason)
        except Exception as e:
            last_err = e
            dprint(f"[JUDGE] ⚠️ attempt {attempt+1} failed: {e}")
    # Both attempts failed → safe conservative default + log it
    dprint(f"[JUDGE] ⚠️ both attempts failed → defaulting grounded (safe)")
    from ._log import warn
    warn("audit_judge", f"malformed/failed judge response: {last_err}")
    return (True, f"judge error after retry (defaulting grounded): {last_err}")
