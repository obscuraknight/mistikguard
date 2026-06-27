# Mistikguard

**Keep your LLM companion's memory honest.**

**Mistikguard** is a small, dependency-light library that stops an LLM's
fabrications from becoming permanent, trusted, defended memory. It is the
extracted memory-integrity core of [Mistik](https://github.com/obscuraknight),
a local-first AI companion, generalized for reuse.

Most "long-term memory" for LLM apps works like this: the model periodically
summarizes the conversation and writes "facts" about the user into a store,
which are then injected back into future prompts as established truth. The flaw
is that the model's output is treated as ground truth the moment it is written.
A hallucinated detail becomes indistinguishable from something the user actually
said — recalled with the same confidence, defended with the same conviction, and
reinforced every time it is injected.

Mistikguard governs that boundary. The model proposes; Mistikguard decides.

---

## The benchmark

The grounding judge — the component that decides whether a memory-claim in a
reply is actually supported — was measured on a 44-case adversarial benchmark
(true recollections, outright fabrications, and deliberately hard borderline
inferences):

| Metric | Value |
|---|---|
| Precision | **1.000** |
| Recall | **0.909** |
| F1 | **0.952** |
| False-positive rate | **0.000** |

The judge is deliberately **safety-biased**: on uncertainty or error it defaults
to *grounded*, so it never raises a false alarm against a true statement. The
price of that 0% false-positive rate is a handful of soft inferences it declines
to flag — a trade chosen on purpose, because for a companion a missed soft
inference costs one sentence, while a false alarm could make the system deny
something real about the user.

The benchmark is reproducible: `python tests/benchmark.py` (needs an
OpenAI-compatible API key). It is a measurement of the **judge in isolation on a
small constructed set**, not a claim about end-to-end fabrication rates in
production. Forty-four cases is modest; treat the figures as indicative.

---

## What it does

Four cooperating pieces:

1. **Provenance.** Every stored fact carries a source — `confirmed` (the user
   stated or ratified it) or `inferred` (the model generated it). Model writes
   default to `inferred`. This single distinction is the keystone everything else
   builds on.

2. **The write-gate.** A deterministic check every model-proposed write must
   pass. It rejects self-narration (the assistant describing its own state),
   contradictions of confirmed fact, and previously-corrected material.

3. **Tombstones.** When the user corrects something, it is *removed* and a
   tombstone is recorded. The gate consults tombstones so corrected material
   cannot be silently re-introduced later. A correction, once made, stays made.

4. **The grounding audit.** After a reply, a cheap pattern detector finds
   memory-claims ("you mentioned…", "I remember that you…"); each is then checked
   by an LLM grounding judge against actual stored memory. Unsupported claims are
   surfaced — not silently rewritten — so a human stays the authority.

The deterministic pieces (provenance, gate, tombstones, the claim detector) have
**zero external dependencies** — pure standard library. Only the grounding judge
needs an LLM client.

---

## Install

```bash
pip install mistikguard          # core only — no external dependencies
pip install mistikguard[llm]     # adds the OpenAI-compatible client for the judge
```

---

## Usage

**Governed facts with provenance:**

```python
from mistikguard.long_memory import LongTermMemory

mem = LongTermMemory(storage_path="./user_memory.json")

# A fact the user stated directly is trusted.
mem.add_fact("User lives in Berlin", source="confirmed")

# A fact the model inferred is held lightly.
mem.add_fact("User probably likes jazz", source="inferred")

# Corrections remove and tombstone — they don't just contradict.
mem.forget_fact("jazz")

print(mem.fact_texts())
```

**The write-gate (deterministic, no API key needed):**

```python
from mistikguard import memory_gate as gate

# Configure once for your assistant and user.
gate.configure(assistant_name="Aria", user_name="Sarah",
               corrections_log_path="./corrections.json")

confirmed = ["User lives in Berlin"]

gate.gate_fact("User lives in Lisbon", confirmed)   # (False, 'contradicts confirmed: ...')
gate.gate_fact("Aria feels calm today", confirmed)  # (False, 'noise/self-narration')
gate.gate_fact("User enjoys hiking", confirmed)     # (True, 'ok')
```

**Auditing a reply for fabricated memory-claims:**

```python
from mistikguard.memory_audit import audit_reply

reply = "Of course — I remember that time we went skydiving together!"
memory_texts = ["User lives in Berlin", "User has a dog named Pixel"]
recent = ["what should I do this weekend?"]

flagged = audit_reply(reply, memory_texts, recent)
# -> [{'phrase': 'i remember that', 'sentence': '... skydiving together!'}]
# Empty list means no fabricated memory-claims were detected.
```

For the LLM-backed judge directly:

```python
from mistikguard.audit_judge import judge_claim
from openai import OpenAI

client = OpenAI(api_key="...", base_url="https://api.groq.com/openai/v1")
grounded, reason = judge_claim(
    client, "model-name",
    "I remember you said your sister is named Olia",
    memory_texts=["User's sister is named Olia"],
    recent_user_msgs=[],
)
# grounded == True
```

---

## What it does *not* do

In the spirit of being honest about its limits:

- **It does not stop the underlying model from producing a false sentence in the
  moment.** That is a property of the model, outside the reach of any surrounding
  structure. What Mistikguard guarantees is narrower and harder: that a false
  sentence does not become *permanent, trusted, defended* memory.
- **The claim detector is pattern-based.** A fabrication phrased in a
  sufficiently novel way can slip the first stage. Coverage is an expanding
  approximation, not a complete solution; the upstream gate is the deeper net.
- **The judge borrows intelligence it does not own.** The grounding decision is
  made by whatever LLM you point it at. Mistikguard's contribution is the
  structure and discipline around the model, not model capability.
- **It is not a content-safety or mental-health tool.** It governs what the
  system treats as true about the user. It is not a clinical resource and not a
  substitute for human or professional support.

The honest description is the useful one: Mistikguard closes the specific failure
that makes most companion memory untrustworthy — model fabrication quietly
becoming durable fact — and is candid about everything it does not close.

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest                 # run the test suite
python tests/benchmark.py        # reproduce the benchmark (needs an API key)
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Mistikguard was originally developed as part of **Mistik**, a local-first AI
companion. See [NOTICE](NOTICE) for attribution.

## Status

Alpha (0.1.0). The core is extracted, generalized, tested, and benchmarked.
APIs may change.
