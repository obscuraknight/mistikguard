import os, json, re
from ._log import dprint


# Configuration: the gate needs the assistant's name (to detect self-narration)
# and the user's name (so the user is never stored as a "person"), plus where to
# keep the corrections/tombstone log. All configurable, generic defaults.
class GateConfig:
    def __init__(self, assistant_name="assistant", user_name="user",
                 corrections_log_path=None):
        self.assistant_name = (assistant_name or "assistant").strip().lower()
        self.user_name = (user_name or "user").strip().lower()
        self.corrections_log_path = corrections_log_path or os.path.join(
            os.getcwd(), "mistikguard_corrections_log.json"
        )

    def noise_patterns(self):
        return [
            rf"^\s*{re.escape(self.assistant_name)}\b",
            r"\bquiet mood\b", r"\bmood drift",
            r"\bcontinuity\b",
            r"\bquiet interval\b", r"\blong absence\b", r"\bafter absence\b",
            r"\bclingy\b", r"\bdramatic\b",
            r"\breflected on\b", r"\bobserved\b.*\bobserved\b",
            r"^\s*a long\b", r"^\s*keep \w+ gently",
        ]

    def non_name_tokens(self):
        base = {
            "you", "i", "me", "my", "your", "user", "the", "a", "an", "she", "he",
            "they", "them", "it", "we", "us", "someone", "somebody", "anyone",
            "person", "people", "friend", "guy", "girl", "man", "woman", "this",
            "that", "here", "there",
        }
        base.add(self.assistant_name)
        base.add(self.user_name)
        return base


_DEFAULT_CONFIG = GateConfig()


def configure(assistant_name="assistant", user_name="user",
              corrections_log_path=None):
    """Set the process-wide default gate configuration. Call once at startup."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = GateConfig(assistant_name, user_name, corrections_log_path)
    return _DEFAULT_CONFIG


def record_correction_tombstone(text, config=None):
    config = config or _DEFAULT_CONFIG
    path = config.corrections_log_path
    try:
        log = {"tombstones": []}
        if os.path.exists(path):
            log = json.load(open(path))
        log.setdefault("tombstones", [])
        t = text.strip().lower()
        if t and t not in log["tombstones"]:
            log["tombstones"].append(t)
            log["tombstones"] = log["tombstones"][-100:]
            json.dump(log, open(path, "w"), indent=2)
            dprint(f"[GATE] 🪦 TOMBSTONE RECORDED: {t[:60]}")
    except Exception as e:
        print("tombstone write error:", e)


def _tombstones(config=None):
    config = config or _DEFAULT_CONFIG
    try:
        if os.path.exists(config.corrections_log_path):
            return json.load(open(config.corrections_log_path)).get("tombstones", [])
    except Exception:
        pass
    return []


def _is_noise(text, config=None):
    config = config or _DEFAULT_CONFIG
    t = text.lower()
    for pat in config.noise_patterns():
        if re.search(pat, t):
            dprint(f"[GATE] 🔇 NOISE DETECTED: pattern '{pat[:30]}...' matched in '{t[:60]}'")
            return True
    if config.assistant_name in t and "user" not in t:
        dprint(f"[GATE] 🔇 NOISE: about the assistant herself, not user")
        return True
    return False


def _contradicts_confirmed(text, confirmed_texts):
    t = text.lower()
    for c in confirmed_texts:
        cl = c.lower()
        m1 = re.search(r"lives in (\w+)", t)
        m2 = re.search(r"lives in (\w+)", cl)
        if m1 and m2 and m1.group(1) != m2.group(1):
            dprint(f"[GATE] ⚠️ CONTRADICTION: location conflict. confirmed='{c[:50]}', proposed='{t[:50]}'")
            return c
        if "does not" in cl or "doesn't" in cl:
            stripped = cl.replace("does not ", "").replace("doesn't ", "")
            key = stripped.replace("user ", "").strip().split()
            if key and key[-1] in t and "not" not in t and "does not" not in t:
                dprint(f"[GATE] ⚠️ CONTRADICTION: negation conflict. confirmed='{c[:50]}', proposed='{t[:50]}'")
                return c
    return None


def gate_fact(text, confirmed_texts, config=None):
    config = config or _DEFAULT_CONFIG
    text = (text or "").strip()
    dprint(f"\n[GATE] ━━━ FACT PROPOSED: '{text[:70]}{'...' if len(text) > 70 else ''}'")
    if not text:
        dprint(f"[GATE] ❌ REJECTED: empty")
        return (False, "empty")
    if _is_noise(text, config):
        dprint(f"[GATE] ❌ REJECTED: noise/self-narration")
        return (False, "noise/self-narration")
    tombs = _tombstones(config)
    if text.strip().lower() in tombs:
        dprint(f"[GATE] ❌ REJECTED: 🪦 tombstoned (recently corrected)")
        return (False, "recently corrected (tombstone)")
    conflict = _contradicts_confirmed(text, confirmed_texts)
    if conflict:
        dprint(f"[GATE] ❌ REJECTED: contradicts confirmed fact")
        return (False, f"contradicts confirmed: {conflict[:60]}")
    dprint(f"[GATE] ✅ ACCEPTED: will store as INFERRED")
    return (True, "ok")


def gate_person(name, relationship="", confirmed_names=None, config=None):
    config = config or _DEFAULT_CONFIG
    confirmed_names = confirmed_names or []
    nm = (name or "").strip()
    dprint(f"\n[GATE] 👤 PERSON PROPOSED: '{nm}'")
    if not nm:
        dprint(f"[GATE] ❌ REJECTED: empty name")
        return (False, "empty name")
    low = nm.lower()
    if low in config.non_name_tokens():
        dprint(f"[GATE] ❌ REJECTED: non-name token")
        return (False, f"non-name token: {nm}")
    if len(nm) < 2 or not any(c.isalpha() for c in nm):
        dprint(f"[GATE] ❌ REJECTED: not a valid name (too short or non-alpha)")
        return (False, f"not a name: {nm}")
    if _is_noise(low, config):
        dprint(f"[GATE] ❌ REJECTED: noise/self-narration")
        return (False, "noise/self-narration")
    tombs = _tombstones(config)
    if low in tombs or f"{low} {relationship}".strip() in tombs:
        dprint(f"[GATE] ❌ REJECTED: 🪦 tombstoned person (recently removed)")
        return (False, "recently removed (tombstone)")
    dprint(f"[GATE] ✅ ACCEPTED: '{nm}' will be stored")
    return (True, "ok")
