# Learning from Experience Without Learning from Noise

This project studies how a tool-using agent can turn past task attempts into reusable instructions without learning the wrong lesson from a misleading failure.

**Project story:** [Read the HTML article](docs/learning-from-experience.html). It uses preliminary, incomplete run data and labels it accordingly.

To view it locally with the chart image, start a small web server from the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Then open `http://localhost:8000/learning-from-experience.html`. To regenerate the article's Matplotlib figures from the saved A0 runs, use `.venv/bin/python docs/generate_article_visuals.py`. Pass `--short-results`, `--long-results`, or `--output-dir` to use other files or an output location.

## Research question

When an agent fails at an API task, the cause may be a fixable weakness in its instructions—or a noisy task, unsupported request, transient environment problem, or bad evaluation signal. If a Reflector treats every failure as evidence about the instructions, its updates can hurt future tasks. We ask whether grounding reflection in request traces, and curating which failures reach the Reflector, improves held-out performance per unit of learning cost.

## Related papers

- **[Agent Plasticity: Measuring Self-Improvement Through Experience](https://arxiv.org/abs/2610.08902)** argues that evaluations of self-improving agents should measure how performance changes on future held-out tasks as the agent learns, and account for the cost of that learning. The paper emphasizes that final capability and learning efficiency can differ, and that gains on training tasks may not fully transfer to held-out tasks.
- **[GEPA-Curator: Curated Learning for Self Improving Agents](https://arxiv.org/abs/2610.37651)** studies two ways to improve reflective prompt optimization: filter experiences likely to provide unreliable learning signals, and retrieve a prior successful trajectory as evidence for the optimizer. Its motivation is that a failed rollout can reflect problems in the prompt, task, tools, environment, or judge.

This repository is a small, controlled study inspired by those ideas; it is **not a reproduction** of either paper's benchmark or reported results.

## What this project implements

The environment is a synthetic REST-style API. Tasks ask an agent to read or change records through API calls. Some tasks are learnable, some are easy controls, and some contain planted noise such as impossible requests or unreliable feedback. The agent carries a playbook forward between training epochs.

Work in this repository has focused on making that comparison inspectable and resumable: the harness records failed requests, error bodies, and later successful requests; validates Reflector rules against those traces; tracks which tasks contributed each playbook rule; and measures both aggregate gate outcomes and held-out performance. A0, A0-raw, and A3 share the actor and task environment so the effect of grounding and curation can be compared. Episodes run concurrently under one rate limiter; transient API failures trigger backoff and a whole-episode retry rather than a scored task failure.

The main comparison is:

| Arm | Behavior |
|---|---|
| `A0` | Reflect on failed tasks, validate proposed rules against request/error traces, then gate a candidate playbook on the current training batch. |
| `A0-raw` | Use the same Reflector prompt and gate, but skip the grounding validator. |
| `A3` | Add a Curator that can defer noisy or well-understood tasks and provide a relevant prior success to the Reflector. Grounding and gating still apply. |

For grounded rules, the validator checks that quoted evidence comes from supplied traces and that the rule names a header, parameter, or field supported by the request or error. Proposed values and formats need trace support, unless the error directly names a missing header (which grounds the header requirement itself). It rejects explicit rules that forbid headers or fields used in supplied successful requests. A candidate rule must pass grounding before the gate runs. The gate compares aggregate scores on the training batch; passing it does not guarantee held-out improvement or prevent an individual task from regressing.

The current hosted-run protocol uses a pool of 20 training tasks, batches of 10, 8 epochs, and 40 held-out in-distribution tasks evaluated at epochs 0, 4, and 8. The task generator plants 20% noise. These are defaults in `scripts/experiments/run_real_experiment.py`; command-line options can shorten a run or select evaluation epochs. Results include held-out success, playbook changes, grounding and gate outcomes, regressions, and per-phase hypothetical model cost.

## Set up

Use Python 3.10 or newer. From the repository root:

```bash
git clone git@github.com:Devansh-Ojha/curation-plasticity.git
cd curation-plasticity
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-phoenix.txt
```

Run the offline test suite. It does not make hosted-model calls:

```bash
python -m pytest -q
```

## Run an experiment

Real runs require an NVIDIA API key and a local Phoenix server. Set the key in your shell; do not put it in `config.yaml` or commit it:

```bash
export NVIDIA_API_KEY="your-key"
phoenix serve
```

Leave Phoenix running in one terminal. In another, activate the environment and start a short A0 run:

```bash
python -m scripts.experiments.run_real_experiment --arm A0 --seed 42 --epochs 3 --eval-epochs 0,3 --run-tag demo
```

For the default 8-epoch protocol, run:

```bash
python -m scripts.experiments.run_real_experiment --arm A0 --seed 42 --run-tag a0-seed42
```

Use `--arm A0-raw` or `--arm A3` to run the other arms. Use seeds `42` or `43`. A new `--run-tag` creates separate local output/log folders and Phoenix experiment names; rerun with the **same** tag to resume that run. Without a tag, the runner uses the standard `seed-<seed>-<arm>` folder and resumes it if a checkpoint exists.

Hosted calls use the actor and optimizer models in `config.yaml`, are subject to provider rate limits, and may incur provider charges. The logged cost and `$1` guard use configured **hypothetical reference pricing**; they are not a provider invoice or exact billing estimate.

## Where files go

- `outputs/real_experiment/seed-<seed>-<arm>-<tag>/` contains checkpoints, generated task data, and `results.json`.
- `logs/real_experiment/seed-<seed>-<arm>-<tag>/calls.jsonl` contains call, retry, and error records. It does not contain the API key.
- Phoenix traces and held-out evaluations are available at `http://localhost:6006` while the local Phoenix server is running.
- `data/` contains the canonical synthetic task files. Ground-truth labels are kept separate from agent prompts.
- `src/` contains the agents, testbed, learning loop, analysis, and observability code. Supporting commands are grouped under `scripts/analysis/`, `scripts/audit/`, and `scripts/legacy/`.

Generated `outputs/`, `logs/`, `checkpoints/`, and `results/` are local artifacts and are excluded from Git. This keeps run data out of the source submission; it does not delete local results.

## Interpreting results

Compare held-out performance at the same checkpoints across arms and seeds, alongside learning cost, rule pollution, gate rejections, and regressions. The gate uses the training batch, so held-out results are the test of generalization. A short run or a single seed is exploratory evidence, not a reliable estimate of an arm's average effect. The repository does not claim final comparative findings until the planned runs are complete and analyzed.

The implementation reports score gain per $1,000 of hypothetical learning cost and also retains raw held-out gain. When preparing a report, include the exact run settings, seed, evaluation checkpoint, and cost label with every result.
