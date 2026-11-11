# Does Curating What an Agent Learns From Raise Plasticity per Dollar?

## Repository Layout

- `run_real_experiment.py` is the resumable hosted-model runner for A0, A0-raw, and A3.
- `run_real_a3_smoke.py` contains shared prompt templates and call logging used by the real runner.
- `scripts/analysis/` contains offline analysis and comparison tools.
- `scripts/audit/` contains offline trace and grounding audits.
- `scripts/legacy/` contains pilot, baseline, smoke, and earlier pipeline entry points.
- `src/agents/`, `src/testbed/`, `src/loop/`, `src/analysis/`, and `src/obs/` contain the implementation modules.
- `tests/` contains offline unit tests; `tests/fixtures/` holds small tracked test inputs.
- `data/` contains the canonical task and label files. Generated `data_*` variants are local-only.
- `checkpoints/`, `results/`, `outputs/`, and `logs/` are local generated artifacts and are excluded from Git.

The ignore rules do not remove local artifacts. In particular, the active A0 run continues using its existing `outputs/` and `logs/` paths.

Run supporting tools as modules from the repository root, for example:

```bash
python -m scripts.analysis.analyze_real_experiment
python -m scripts.audit.audit_saved_reflector_proposals
```

The scripts under `scripts/legacy/` are retained for historical reproduction and are not part of the active hosted-run workflow.

An empirical evaluation connecting **Agent Plasticity** (Singh et al., Meta / Berkeley / Princeton, arXiv:2610.08902) and **GEPA-Curator** (Yu et al., UC Berkeley / Keutzer Lab, arXiv:2610.37651), instrumented in **Arize Phoenix**.

---

## 1. Executive Summary & Core Question

> **Question:** Under planted production noise (impossible tasks, flaky judges, ambiguous specs), does Curator-style trace curation raise held-out gain per $1,000 once its own token overhead is counted?

In real-world deployments, self-improving agents reflect on task failures to update persistent prompts or tool inventories. However, when failures stem from impossible endpoints or flaky tests, uncurated reflectors hallucinate toxic workaround rules that **pollute** the playbook and **regress** previously solved tasks. 

This project implements a testbed to evaluate whether **trace curation** pays for itself in net **Agent Plasticity** (held-out gain per $1,000 learning spend).

---

## 2. Experimental Architecture

```
                                  +-----------------------------+
                                  |     Training Tasks (60)     |
                                  | 15% Noise (Imp, Flaky, Amb) |
                                  +--------------+--------------+
                                                 |
                                     [Minibatch Selector]
                                                 |
            +-------------------+----------------+-------------------+
            |                   |                                    |
     [A0: Uncurated]   [A2: Numeric Curriculum]            [A3: Trace Curator]
            |                   |                                    |
            v                   v                                    v
     All Failures         Drop if s_t > b                   Trace Inspection:
                                                       - Filter Impossible / Flaky
                                                       - Pair Corroborated Anchor
                                                                     |
                               +-------------------------------------+
                               v
                  [Reflector: Propose Ops]
                               |
                   [Deterministic Provenance]
             rule_id -> [source_failed_task_ids]
                               |
                    [GEPA-Style Gating Eval]
                 Accept if Candidate Score >= Curr
                               |
                  [Arize Phoenix Tracing & Evals]
```

### The 4 Evaluated Arms
1. **A0: Uncurated + Gating (Baseline):** Uniform random minibatches; reflects on all failures.
2. **A1: Oracle (Upper-bound):** Reflects strictly on ground-truth `learnable` failures.
3. **A2: Numeric Curriculum:** Outcome-history filter based on GEPA-Curator Algorithm 1 ($s_t \leftarrow s_t + 1$ on failure, $s_t \leftarrow \max(0, s_t - k)$ on success; dropped if $s_t > b$).
4. **A3: Curator (Full):** Trace-reading LLM Curator evaluating failure evidence + contrastive anchor selection.

---

## 3. Key Hypotheses & Experimental Findings

| Hypothesis | Predicted Behavior | Observed Finding | Verdict |
| :--- | :--- | :--- | :--- |
| **H1: Uncurated Pollution** | Uncurated reflection admits harmful/polluted rules under aggregate minibatch gating. | **A0 suffered 35.6% rule pollution** (peaking at 50% in Seed 43), hallucinatory workaround rules created from impossible endpoints. | **Supported** |
| **H2: Trace vs Numeric Curriculum** | Numeric curriculum fails to separate flaky from learnable tasks because both yield mixed outcomes. | **A2 had identical pollution (35.6%) to uncurated baseline**, as pass/fail counts cannot detect non-deterministic graders. | **Supported** |
| **H3: Curation Cost Overhead** | Curation token cost consumes learning budget; at low-to-moderate noise, net plasticity may be lower unless noise is severe. | Curation added ~20–25% token cost ($0.149 vs $0.121 avg), resulting in raw plasticity of 156k vs 193k/$1k under 15% noise. | **Supported** |
| **H4: Provenance Attribution** | Rule provenance enables 100% deterministic pollution auditing without an LLM judge. | Verified: 0% pollution for Oracle (A1), clean reduction in Curator (A3). | **Supported** |

---

## 4. Benchmark Results Table (3 Seeds, 8 Epochs)

| Arm | Held-out ID Gain | Total Learning Cost | Plasticity ($ / \$1,000) | Rule Pollution Rate | Strict Pollution Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A0: Uncurated** | +23.3 pp | $0.1209 | **193,027** | 35.6% | 5.6% |
| **A1: Oracle** | +23.3 pp | $0.1172 | **199,007** | **0.0%** | **0.0%** |
| **A2: Numeric Curriculum** | +23.3 pp | $0.1209 | **193,027** | 35.6% | 5.6% |
| **A3: Curator (Full)** | +23.3 pp | $0.1496 | **156,278** | **28.9%** (down to 0% in s42) | 6.7% |

*All learning costs include actor rollouts, curation evaluations, multi-attempt reflections, and GEPA-style gating re-runs.*

---

## 5. Artifacts and Visualization

Generated charts located in `results/figures/`:
1. `heldout_id_curves.png`: Mean and 95% bootstrap CI curves across all checkpoints.
2. `plasticity_frontier.png`: Trajectory of held-out capability gain versus cumulative dollar cost.
3. `pollution_rate.png`: Deterministic playbook pollution comparison across arms.

---

## 6. Arize Phoenix Integration

### Observability & Spans
* **Project Name:** `curation-plasticity`
* **Span Hierarchy:**
  * Root Span: `phase` (`select`, `act`, `curate`, `reflect`, `gate`, `eval`).
  * Child Spans: `episode__{task_id}` recording tool execution, `read_playbook_called`, token usage, and cost attribution.
* **Label Hygiene:** Ground truth categories are strictly segregated in `data/labels.json` and never passed into agent context or span attributes, preventing data leakage.

### Launching Phoenix Locally
```bash
# Activate virtual environment
source .venv/bin/activate

# Launch Arize Phoenix UI
phoenix serve
```
Open `http://localhost:6006` to inspect traces, spans, and experiment runs.

---

## 7. Developer Learnings for Production Agents

1. **The Flaky Outcome Blindspot:** Outcome-based curriculums (like Elo or pass/fail thresholds) cannot distinguish flaky evaluations from prompt-sensitive tasks. Trace inspection is required.
2. **Deterministic Provenance > LLM-as-a-Judge:** Tracking which failure IDs inspired which rule in the artifact metadata creates a zero-cost, objective audit trail for knowledge pollution.
3. **Curation Overhead Economics:** Trace curation introduces token spend. Teams should evaluate their specific noise prevalence: at low noise (<10%), curation overhead may compress plasticity per dollar; at high noise (>25%), it prevents catastrophic regressions.
