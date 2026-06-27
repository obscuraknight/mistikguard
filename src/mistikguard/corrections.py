import json
from ._log import dprint

# Trigger phrases that signal the user may be correcting something.
_TRIGGERS = (
    "that's wrong", "thats wrong", "that is wrong", "that's not right", "thats not right",
    "not correct", "incorrect", "that's a mistake", "thats a mistake", "is a mistake",
    "forget that", "forget about", "forget it", "delete that", "delete it", "remove that",
    "i never said", "i didn't say", "i did not say", "i never told you",
    "where did you get", "i don't", "i do not", "i'm not", "i am not",
    "not true", "that's false", "stop saying", "i said", "actually it's", "actually its",
    "wrong about", "you're wrong", "youre wrong", "you are wrong", "no i work", "i work in", "i work at",
)

def detect_correction(message: str) -> bool:
    """Cheap deterministic check — returns True if the message looks like it may
    correct a stored fact. Runs on every message; only when True do we make the
    (more expensive) extraction LLM call."""
    if not message:
        return False
    m = message.lower()
    triggered = any(t in m for t in _TRIGGERS)
    if triggered:
        dprint(f"[CORRECT] 🚨 CORRECTION DETECTED: '{message[:80]}'")
    return triggered

def extract_correction(client, model, message: str, fact_texts: list) -> dict:
    """Called only when detect_correction fired. Focused LLM call: given the current
    stored facts and the user's message, identify which facts to DELETE and what
    corrected truth (if any) to store. The LLM does NOT execute — it returns a plan.
    Returns {"delete": [substrings], "add_confirmed": [new fact strings]}."""
    dprint(f"\n[CORRECT] ━━━━ EXTRACTING CORRECTION")
    facts_block = "\n".join(f"{i}. {t}" for i, t in enumerate(fact_texts)) or "(no facts stored)"
    prompt = (
        "The user may be correcting something the AI companion wrongly believes about them. "
        "Below are the facts currently stored about the user, and the user's latest message.\n\n"
        "Decide:\n"
        "1. Which stored fact(s), if any, are now contradicted or denied by the user and must be DELETED. "
        "For each, give a short distinctive substring that matches that fact (enough to identify it).\n"
        "2. What corrected TRUTH, if any, the user is stating that should be stored as a confirmed fact. "
        "Write it as a short standalone statement (e.g. 'User works at TechCorp').\n\n"
        "If the user is NOT correcting a fact (just chatting), return empty lists.\n"
        "If the user denies knowing a person or says someone is not their relative/friend "
        "(e.g. 'that's not my sister', 'I don't know anyone named X', 'forget about X'), "
        "put that person's NAME in remove_people.\n"
        "Be conservative: only delete a fact if the user clearly contradicts or denies it. "
        "Only add a fact if the user clearly states a correcting truth.\n\n"
        "Reply ONLY as JSON, no markdown:\n"
        '{"delete": ["substring1"], "add_confirmed": ["corrected truth"], "remove_people": ["PersonName"]}\n\n'
        f"STORED FACTS:\n{facts_block}\n\n"
        f"USER MESSAGE:\n{message}"
    )
    try:
        dprint(f"[CORRECT] Calling LLM to extract correction plan...")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        result = {
            "delete": [s for s in data.get("delete", []) if isinstance(s, str) and s.strip()],
            "add_confirmed": [s for s in data.get("add_confirmed", []) if isinstance(s, str) and s.strip()],
            "remove_people": [s for s in data.get("remove_people", []) if isinstance(s, str) and s.strip()],
        }
        if result["delete"]:
            dprint(f"[CORRECT] 🗑️  TO DELETE: {result['delete']}")
        if result["add_confirmed"]:
            dprint(f"[CORRECT] ➕ TO ADD (CONFIRMED): {result['add_confirmed']}")
        if result["remove_people"]:
            dprint(f"[CORRECT] 👤 TO REMOVE: {result['remove_people']}")
        if not any([result["delete"], result["add_confirmed"], result["remove_people"]]):
            dprint(f"[CORRECT] ✓ No corrections identified")
        return result
    except Exception as e:
        print(f"[CORRECT] extract_correction error: {e}")
        from ._log import warn
        warn("extract_correction", str(e))
        return {"delete": [], "add_confirmed": [], "remove_people": []}
