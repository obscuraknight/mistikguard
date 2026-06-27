import os
import json
import datetime
from .storage import safe_save_json

# ─────────────────────────────────────────────
#  LONG-TERM MEMORY
# ─────────────────────────────────────────────
_DEFAULT_LONG_MEMORY_FILE = os.path.join(os.getcwd(), "mistikguard_longmem.json")
CURRENT_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class LongTermMemory:
    """Persists facts, the user's name, and a rolling summary across sessions.

    Facts are stored as dicts:
        {"text": str, "source": "confirmed"|"inferred", "added_at": ISO-str}

    Old bare-string facts are auto-migrated on load using _PRIORITY_KEYWORDS
    as a heuristic: personal/relational facts → confirmed, rest → inferred.
    """

    def __init__(self, storage_path=None):
        self.path = storage_path or _DEFAULT_LONG_MEMORY_FILE
        self.data = {
            "version":   CURRENT_SCHEMA_VERSION,
            "user_name": None,
            "facts":     [],   # list of {"text", "source", "added_at"} dicts
            "summary":   "",
            "sessions":  0,
        }
        self._load()

    # Keywords that signal a fact names a real person, place, or role —
    # used both for priority-sorting in get_context() and migration heuristic.
    _PRIORITY_KEYWORDS = (
        "partner", "sister", "brother", "mother", "father", "friend", "wife",
        "husband", "girlfriend", "boyfriend", "colleague", "boss", "child",
        "daughter", "son", "named", "name is", "lives in", "from ", "works at",
        "works on", "job is", "studying",
    )

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.data.update(json.load(f))
            except Exception:
                pass
        # ── Migrate bare strings → dicts ──────────────────────────────────────
        migrated = []
        needs_save = False
        for entry in self.data.get("facts", []):
            if isinstance(entry, dict):
                migrated.append(entry)
            else:
                needs_save = True
                text = str(entry)
                kw   = self._PRIORITY_KEYWORDS
                src  = "confirmed" if any(k in text.lower() for k in kw) else "inferred"
                migrated.append({"text": text, "source": src, "added_at": _now()})
        self.data["facts"] = migrated
        # Stamp schema version on legacy files that predate versioning.
        if self.data.get("version") != CURRENT_SCHEMA_VERSION:
            self.data["version"] = CURRENT_SCHEMA_VERSION
            needs_save = True
        if needs_save:
            self.save()

    def save(self):
        safe_save_json(self.path, self.data)

    def set_name(self, name):
        self.data["user_name"] = name
        self.save()

    def add_fact(self, text, source="inferred"):
        """Add a fact with provenance.

        source="inferred"  — LLM-extracted guess (default, held lightly)
        source="confirmed" — user stated/approved (trusted, durable)

        Deduplication: if an identical text already exists and the new source
        is "confirmed" while the stored one is "inferred", promote it.
        Exact duplicates with no promotion needed are silently dropped.
        """
        text = text.strip()
        if not text:
            return
        for entry in self.data["facts"]:
            if entry["text"] == text:
                # Promote inferred → confirmed if caller now confirms it
                if source == "confirmed" and entry["source"] == "inferred":
                    entry["source"] = "confirmed"
                    self.save()
                return
        self.data["facts"].append({"text": text, "source": source, "added_at": _now()})
        self.data["facts"] = self.data["facts"][-60:]
        self.save()

    def remember_fact(self, text: str) -> None:
        """Manually seed a fact the user explicitly stated — always confirmed."""
        self.add_fact(text.strip(), source="confirmed")

    def confirm_fact(self, substring: str) -> int:
        """Promote every fact whose text contains `substring` to confirmed.
        Returns the count of facts promoted."""
        needle   = substring.lower()
        promoted = 0
        for entry in self.data["facts"]:
            if needle in entry["text"].lower() and entry["source"] != "confirmed":
                entry["source"] = "confirmed"
                promoted += 1
        if promoted:
            self.save()
        return promoted

    def forget_fact(self, substring: str) -> int:
        """Remove every fact whose text contains `substring` (case-insensitive).
        Returns the number of facts removed."""
        needle  = substring.lower()
        before  = self.data["facts"][:]
        kept    = [f for f in before if needle not in f["text"].lower()]
        removed = len(before) - len(kept)
        if removed:
            self.data["facts"] = kept
            self.save()
        return removed

    def fact_texts(self):
        """Return facts as plain text strings, for callers that just display/scan text."""
        return [f["text"] if isinstance(f, dict) else f for f in self.data["facts"]]

    def update_summary(self, new_summary):
        self.data["summary"] = new_summary
        self.data["sessions"] += 1
        self.save()

    def get_context(self):
        """Return a formatted string to inject into the system prompt.
        Facts are split by provenance: confirmed (state as fact) vs inferred
        (hold lightly, signal uncertainty). Phase 4."""
        parts = []
        if self.data["user_name"]:
            parts.append(f"The user's name is {self.data['user_name']}.")
        if self.data["summary"]:
            parts.append(f"Summary of past sessions: {self.data['summary']}")

        def _priority_sorted(group):
            kw = self._PRIORITY_KEYWORDS
            pri  = [f for f in group if any(k in f["text"].lower() for k in kw)]
            rest = [f for f in group if not any(k in f["text"].lower() for k in kw)]
            return pri + rest

        confirmed = [f for f in self.data["facts"] if f.get("source") == "confirmed"]
        inferred  = [f for f in self.data["facts"] if f.get("source") != "confirmed"]

        if confirmed:
            block = "\n".join(f"- {f['text']}" for f in _priority_sorted(confirmed)[:30])
            parts.append(
                "What you KNOW about this person (confirmed — they told you or approved it; "
                "state these plainly as fact):\n" + block
            )
        if inferred:
            block = "\n".join(f"- {f['text']}" for f in _priority_sorted(inferred)[:20])
            parts.append(
                "What you've INFERRED but they have NOT confirmed (hold these lightly — "
                "voice them with 'I think', 'I have the impression', 'unless I'm wrong'; "
                "invite correction, and never assert them as certain):\n" + block
            )
        return "\n\n".join(parts) if parts else ""

    def extract_facts_from_conversation(self, client, model, history):
        """Ask the AI to extract concrete, specific facts from the conversation.
        Extracted facts default to source='inferred' — they are LLM guesses."""
        if len(history) < 4:
            return
        convo = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in history[-20:]
            if isinstance(m.get("content"), str)
        )
        prompt = (
            "Extract SPECIFIC, CONCRETE facts about the user that a companion must "
            "remember long-term. Capture: names of people and their relationship to "
            "the user (e.g. 'User\\'s partner is Franzi', 'User\\'s friend Stavros'), "
            "places they live or are from, their job/projects with specifics, concrete "
            "commitments, and specific stable preferences. Write each fact as a short "
            "standalone statement that includes the actual name or detail.\n\n"
            "DO NOT write vague mood or theme summaries like 'enjoys meaningful "
            "conversations' or 'concerned about suffering' — those are useless.\n\n"
            "If the conversation contains a person\\'s name and how they relate to the "
            "user, ALWAYS capture it. Prefer 'User\\'s sister is named X' over 'User "
            "talked about family'.\n\n"
            "GOOD: \"User's partner is Franzi\", \"User lives in Berlin\", "
            "\"User is building a modular Python web app\"\n"
            "BAD: \"User has meaningful relationships\", \"User values connection\"\n\n"
            "Also write a 1-2 sentence session summary of WHAT HAPPENED THIS SESSION "
            "(concrete events/actions only, e.g. 'Worked on the audit system and tested "
            "fabrication detection'). The summary must NOT assert traits, preferences, or "
            "facts about the user (e.g. NOT 'User's favorite color is teal') — those belong "
            "only in the facts list. Episodic events only.\n\n"
            "Reply ONLY as JSON — no markdown fences:\n"
            '{"facts": ["fact1", "fact2"], "summary": "...", "user_name": "name or null"}'
            "\n\nCONVERSATION:\n" + convo
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            try:
                from . import memory_gate as _gate
                confirmed_texts = [f["text"] for f in self.data["facts"] if isinstance(f, dict) and f.get("source") == "confirmed"]
            except Exception:
                _gate = None; confirmed_texts = []
            for fact in data.get("facts", []):
                if _gate:
                    allowed, reason = _gate.gate_fact(fact, confirmed_texts)
                    if not allowed:
                        print(f"[gate] extracted fact rejected ({reason}): {str(fact)[:50]}")
                        continue
                self.add_fact(fact)          # source defaults to "inferred" — correct
            if data.get("summary"):
                summ = data["summary"]
                # gate the summary too: reject self-narration noise so it can't
                # launder ungoverned claims into context
                if _gate:
                    is_noise = False
                    try:
                        is_noise = _gate._is_noise(summ)
                    except Exception:
                        is_noise = False
                    if is_noise:
                        print(f"[gate] summary rejected (noise): {summ[:50]}")
                        summ = ""
                if summ:
                    self.update_summary(summ)
            if data.get("user_name"):
                self.set_name(data["user_name"])
        except Exception as e:
            print("Memory extraction error:", e)
