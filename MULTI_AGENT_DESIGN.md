# Multi-Agent System Design for AgentClinic

## Dataset Overview

| Dataset | Scenarios | Format | Key Characteristics |
|---------|-----------|--------|---------------------|
| **MedQA** | 107 | OSCE (History, Exam, Tests, Correct_Diagnosis) | Full case upfront; doctor asks questions; tests available on request |
| **MedQA_Ext** | 214 | Same as MedQA | 182 unique diagnoses; high diversity |
| **NEJM** | 14 | Question + Image + patient_info + physical_exams + multiple-choice answers | Image-heavy; answer choices constrain space |
| **NEJM_Ext** | 119 | Same as NEJM | Case reports with images |
| **MIMIC-IV** | (requires PhysioNet approval) | Real clinical cases | Different distribution |

**Total: ~450+ scenarios** across specialties (neuro, GI, derm, rheum, cardio, psych, peds, etc.)

---

## Current Architecture (What You Have)

```
Patient Agent ←→ Doctor Agent
       ↑              ↑
       └── Question Controller (suggests next question)
       └── Measurement Agent (returns test results on REQUEST TEST)
       └── Reasoning Critic + Evidence Checker (on first DIAGNOSIS READY)
       └── DX Normalizer (before grading)
       └── Moderator (grades match)
```

**Failure modes observed:**
1. **Information gathering**: Doctor asks low-yield questions, runs out of turns before key evidence
2. **Differential narrowing**: Doctor locks on wrong diagnosis early, doesn't reconsider
3. **Similar-condition confusion**: Picks wrong member of a pair (Pes anserine vs meniscus, IBS vs UC, etc.)
4. **Normalizer corruption**: LLM normalizer sometimes substitutes wrong diagnosis
5. **Single-shot reasoning**: One critic + one revision may not be enough for hard cases

---

## Proposed Multi-Agent Architecture (No Training, No Clues)

### Core Principle
Improve accuracy through **structure and redundancy**, not through prompt hints. The system should reason better by having more agents reason, debate, and refine—using only their base medical knowledge.

---

### 1. **Differential-First Doctor**

**Change**: Require the doctor to output a **ranked differential (top 3–5)** every N turns, and only accept `DIAGNOSIS READY` if it appears in the differential.

**Why**: Forces considered reasoning. Prevents random guesses. The doctor must maintain and update a differential, not jump to a conclusion.

**Implementation**: Add a `working_ddx` output block (you already have this in STATE_JSON). Enforce: "Before DIAGNOSIS READY, your final diagnosis must be one of your top 3 in working_ddx."

---

### 2. **Multi-Round Adversarial Critique**

**Change**: Instead of one Reasoning Critic + one Evidence Checker, run **2–3 rounds** of critique:

- **Round 1**: Critic suggests alternatives + distinction; Evidence Checker gives SUPPORT/CONTRADICT/ABSENT
- **Doctor revises** (or confirms)
- **Round 2**: If doctor confirms, run a **"Devil's Advocate"** agent: "Argue against this diagnosis. What is the strongest reason it could be wrong?"
- **Doctor finalizes**

**Why**: More opportunities to catch errors. The devil's advocate forces steelmanning the counter-argument without giving the answer.

---

### 3. **Specialist Ensemble (Optional)**

**Change**: For complex cases, run 2–3 "specialist" agents in parallel (e.g., generalist, plus one focused on the chief complaint domain). Each proposes a diagnosis. A **Synthesizer** agent sees all proposals + evidence and picks one (or flags disagreement).

**Why**: Different "perspectives" can catch what a single agent misses. No hints—just different system prompts ("You are a neurologist" vs "You are a general internist").

**Caveat**: 2–3x API cost per case. Use only when confidence is low or case is complex.

---

### 4. **Question Quality Scorer**

**Change**: After each doctor question, a lightweight agent scores: "How discriminative was this question for the differential?" (1–5). If score is low for 2+ consecutive turns, inject: "Consider asking a more discriminative question—one that rules in/out specific diagnoses."

**Why**: Improves information gathering without telling the doctor what to ask. Purely procedural feedback.

---

### 5. **Test-Ordering Agent**

**Change**: Separate **test-ordering** from diagnosis. A dedicated agent proposes: "Given the differential [X, Y, Z], what single test would best distinguish them?" Doctor can accept or override.

**Why**: Doctors sometimes don't request the right tests. A focused agent might do better. No clues—it only sees the differential and case so far.

---

### 6. **Uncertainty-Gated Commitment**

**Change**: Doctor must output a confidence level with DIAGNOSIS READY (e.g., "DIAGNOSIS READY: X [confidence: high/medium/low]"). If low, automatically trigger an extra reasoning round (critic + devil's advocate) before grading.

**Why**: Low-confidence diagnoses are more likely wrong. One more round could flip them.

---

### 7. **Conservative Normalizer**

**Change**: Make the normalizer **much more conservative**:
- Only rephrase (e.g., "Crohn's disease" → "Crohn disease")
- Never substitute a different condition
- If the LLM suggests a substitution, reject it and keep the original

**Why**: The normalizer has been corrupting correct diagnoses (Dumping→Hypoglycemia, etc.). A strict "no substitution" policy prevents that.

---

### 8. **Dataset-Adaptive Flow**

**Change**: Detect dataset type and adjust:

| Dataset | Adaptation |
|---------|------------|
| **NEJM/NEJM_Ext** | Pass answer choices to doctor (they're in the case); reduce question budget if image is highly informative |
| **MedQA/MedQA_Ext** | Full 20-turn history-taking; emphasize test-ordering when exam is ambiguous |
| **MIMIC-IV** | May have longer histories; consider summarization for context window |

**Why**: Different datasets have different information structures. Adapting the flow (not the prompts) can help.

---

### 9. **Remove Remaining Hints**

**Change**: The Question Controller prompt contains diagnostic-specific hints (e.g., "For rectal bleeding: Ask about mass characteristics..."). Replace with a **fully generic** prompt: "Propose the single most discriminative question given the transcript and differential. No specific disease guidance."

**Why**: Aligns with "no clues" — the system should reason from first principles.

---

## Recommended Implementation Order

1. **Conservative Normalizer** (quick win, prevents corruption)
2. **Differential-First Enforcement** (structural, no new agents)
3. **Multi-Round Critique + Devil's Advocate** (2nd round of reasoning)
4. **Question Quality Scorer** (improves information gathering)
5. **Uncertainty-Gated Extra Round** (targets low-confidence cases)
6. **Test-Ordering Agent** (separate concern)
7. **Specialist Ensemble** (optional, for hardest cases)
8. **Dataset-Adaptive Flow** (polish)

---

## Expected Impact

- **Normalizer fix**: Prevents ~5–10% of errors (correct diagnoses being corrupted)
- **Multi-round critique**: +5–15% on hard cases (more chances to revise)
- **Differential-first**: +3–8% (reduces impulsive wrong commits)
- **Question quality**: +2–5% (better evidence gathered)
- **Combined**: Realistic target of **+15–25%** over baseline without any training or diagnostic hints

---

## What This Does NOT Do

- No fine-tuning or training
- No scenario-specific or diagnosis-specific prompt hints
- No access to correct answer during reasoning
- No retrieval of case-specific guidance (RAG could be added with generic guidelines only)

---

## Files to Modify

| Component | File | Changes |
|-----------|------|---------|
| Normalizer | `agentclinic_enhanced.py` | Add strict no-substitution policy |
| Doctor | `agentclinic_enhanced.py` | Enforce differential-first; add confidence output |
| Main loop | `agentclinic_enhanced.py` | Multi-round critique; devil's advocate; uncertainty gate |
| Question Controller | `agentclinic_enhanced.py` | Generic prompt (remove diagnostic hints) |
| New agents | `agentclinic_enhanced.py` | QuestionQualityScorer, TestOrderingAgent, DevilAdvocate |
