# Can LLMs Write Correct TLA+ Specifications?

## Presentation Summary -- Nontechnical Audience

**Lab:** AI4FM Research Group, Loyola University Chicago Department of Computer Science\
**Mission:** Advance formal methods, rigorous system design, and reproducible tools at the intersection of AI, logic, mathematics, and computing\
**Faculty:** Konstantin Laufer, Mohammed Abuhamad, George K. Thiruvathukal, TaiNing Wang\
**Status:** Under submission

---

## Slide 1 -- Introduction

Software bugs in critical systems -- cloud infrastructure, banking, medical devices -- can
cause outages, data loss, or worse. Companies like Amazon and Microsoft use a mathematical
language called **TLA+** to *prove* their systems are correct before they ever run in
production. The problem is that writing TLA+ requires specialized expertise that most
engineers do not have.

**Our question:** Can we teach an AI to write TLA+ for us?

This research comes out of the **AI4FM lab at Loyola University Chicago**, a group focused
on making formal verification more accessible through AI. We evaluated 30 large language
models to find out whether any of them can reliably translate plain-English descriptions of
a system into correct TLA+ specifications -- and when they fail, *why* they fail.

---

## Slide 2 -- What Is TLA+?

TLA+ is a formal specification language for describing how systems behave over time. Think
of it as a blueprint that a computer can automatically check for flaws.

**Key concepts in plain language:**

| TLA+ Concept | Everyday Analogy |
|---|---|
| **Specification** | A precise set of rules describing how a system is allowed to behave |
| **State** | A snapshot of the system at one moment (e.g., "the door is locked") |
| **Init** | The starting snapshot (e.g., "the door starts locked") |
| **Next** | The allowed moves from one snapshot to the next |
| **Invariant** | A safety rule that must *always* be true (e.g., "the door is never open and unlocked at the same time") |
| **SANY** | A syntax checker -- "Is this valid TLA+?" |
| **TLC** | A model checker -- "Does this specification actually satisfy its safety rules?" |

**Why it matters:** Amazon used TLA+ to find critical bugs in DynamoDB and S3 that
conventional testing missed entirely. Microsoft Azure uses it for distributed protocols.
If AI could generate TLA+ from English, formal verification would be accessible to any
engineer, not just specialists.

---

## Slide 3 -- Basic Models, Reasoning Models, and Prover Models

Not all AI models work the same way. We tested models across three broad categories:

### Basic Models (e.g., LLaMA, Mistral, Gemma)
- Trained on large amounts of text and code from the internet
- Good at pattern matching and completing familiar-looking code
- **Limitation:** TLA+ looks very different from Python or Java, so these models
  frequently produce syntactically broken output

### Reasoning Models (e.g., DeepSeek R1, QwQ)
- Designed to "think step-by-step" before answering
- Better at logical structure and multi-step problems
- **Our finding:** These performed best overall; DeepSeek R1 at only 8 billion parameters
  outperformed models 9x its size

### Code-Specialized Models (e.g., Qwen-Coder, CodeLLaMA)
- Specifically trained on programming languages (Python, JavaScript, etc.)
- **Surprising finding:** These actually performed *worse* on TLA+. Being good at
  mainstream programming caused **negative transfer** -- the models tried to write TLA+
  as if it were Python, producing broken specifications

**Takeaway:** For formal languages, *how* a model reasons matters more than how large it
is or how much code it has seen.

---

## Slide 4 -- Current Best Models for TLA+

We evaluated 30 models (25 open-weight, 5 proprietary) on a curated dataset of 205 TLA+
specifications, using four prompting strategies.

### Top-line results

| Metric | Best Result |
|---|---|
| Syntactic correctness (passes SANY) | 26.6% |
| Semantic correctness (passes TLC) | 8.6% |

### What the prompting strategies mean

| Strategy | Description | Outcome |
|---|---|---|
| **Direct** | "Write a TLA+ spec for X" | Mostly garbage |
| **Few-shot** | Show 2-3 examples first, then ask | Better syntax, no semantic success |
| **Chain-of-thought** | Ask the model to reason step-by-step | Improved structure |
| **Progressive** | Build the spec piece by piece (Init, then Next, then Invariant) | **Only strategy that achieved semantic correctness** |

### Key surprises
- **Size does not equal quality.** DeepSeek R1 8B beat DeepSeek R1 70B across every
  prompting strategy.
- **Code training hurts.** Models trained heavily on mainstream code consistently
  underperformed general or reasoning-focused models.
- **Semantic success requires progressive prompting.** No model produced a TLC-valid
  specification under direct or few-shot prompting alone.

---

## Slide 5 -- How to Create a Better Model via Fine-Tuning

Since off-the-shelf models top out at 8.6% correctness, we are building **ChatTLA** -- a
model fine-tuned specifically for TLA+ generation with a self-improving training loop.

### Our approach

1. **Curate training data**
   - Start with 205 expert-written TLA+ specs from the FormaLLM dataset
   - Scrape and validate additional specs from GitHub
   - Every spec is checked by SANY (syntax) and TLC (semantics) before use

2. **Fine-tune with LoRA**
   - Take an existing 20-billion-parameter model (GPT-OSS 20B)
   - Apply *Low-Rank Adaptation* -- a technique that trains a small number of extra
     parameters on top of the frozen base model
   - This is fast and resource-efficient compared to training from scratch

3. **Use the model checker as the training signal**
   - Traditional AI training optimizes for "does the output look right?"
   - We optimize for "does the output *pass the model checker*?"
   - A generated spec only counts as correct if TLC says it has zero violations

4. **Autonomous self-improvement loop (RL)**
   - The model generates new specs from prompts
   - TLC validates each one: correct specs become new training data
   - The model retrains on its own successes, getting better over time
   - This runs continuously on two GPUs (Quadro RTX 8000, 96 GB total VRAM)

### Why this should work
- The model checker provides a **perfect, automatic grading signal** -- unlike most AI
  tasks where correctness is subjective
- The self-improvement loop means the training dataset grows with each cycle
- Fine-tuning on TLA+ specifically addresses the negative transfer problem we observed
  in code-specialized models

---

## Slide 6 -- Conclusion

### What we showed
- Current LLMs **cannot reliably generate correct TLA+ specifications** -- the best
  model achieves only 8.6% semantic correctness
- **Model size does not predict quality** -- a small reasoning model beat one 9x larger
- **Code specialization hurts** -- mainstream programming knowledge causes negative
  transfer to formal languages
- **Progressive prompting is essential** -- it is the only strategy that produces
  semantically valid output

### What we are building
- **ChatTLA** is a fine-tuned model that uses the TLC model checker itself as the
  training signal, with an autonomous self-improvement loop
- The goal: make formal verification accessible to engineers who understand their systems
  but lack TLA+ expertise

### Why it matters
- Formal verification catches bugs that testing cannot
- It is already used at Amazon, Microsoft, and other companies for mission-critical systems
- If AI can bridge the gap between English and TLA+, formal methods move from specialist
  tooling to mainstream engineering practice

---

## About AI4FM

The **AI4FM** (AI for Formal Methods) research group at **Loyola University Chicago**
works at the intersection of artificial intelligence, formal methods, and software
engineering. Research areas include:

- **Formal Specification and Verification** -- Making TLA+ and model checking accessible
  through open pipelines and notebook workflows
- **LLMs for Formal Methods** -- Evaluating and improving AI-generated formal specifications
- **Empirical Software Engineering** -- Studying software artifacts and development practices
- **Security and Systems** -- IoT security, signal injection attacks, HPC education

**Tools:** [TLA+ for All](https://ai4fm.cs.luc.edu/) -- a Python notebook environment for
model checking without installation

**Website:** <https://ai4fm.cs.luc.edu/>
