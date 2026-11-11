# Learning from Experience Without Learning from Noise

## The project

This project studies whether an API agent can learn useful, reusable instructions from past attempts without treating every failure as a trustworthy lesson. Failures can come from a fixable prompt weakness, but also from a noisy task, an unsupported request, a transient service problem, or an evaluation mistake.

The agent works in a synthetic REST API environment and carries a playbook between training epochs. The experiment compares three approaches:

- **A0:** reflect on failed tasks, check proposed rules against request and error traces, then evaluate the candidate playbook on the training batch.
- **A0-raw:** use the same Reflector and gate, but skip trace grounding.
- **A3:** add task curation and prior-success retrieval before reflection; keep grounding and gating.

The question is whether grounding and curation improve learning on held-out tasks and how much model usage that learning takes. This small study is inspired by [Agent Plasticity](https://arxiv.org/abs/2610.08902) and [GEPA-Curator](https://arxiv.org/abs/2610.37651); it does not reproduce their benchmarks.

## Get a free NVIDIA API key

1. Sign in or create an NVIDIA account at the [NVIDIA API Catalog](https://build.nvidia.com/explore).
2. Open a model page labeled **Free Endpoint** and select **Get API Key**. Follow the sign-in and developer program prompts. NVIDIA displays the new key after setup; copy it and store it somewhere private. See NVIDIA’s [API Catalog quickstart](https://docs.api.nvidia.com/nim/docs/api-quickstart).
3. Check the selected model’s page for its current access terms and limits. Free endpoint availability, quotas, and rate limits can vary and change.
4. In the terminal where you will run the project, provide the key as an environment variable:

   ```bash
   printf "NVIDIA API key: "
   read -s NVIDIA_API_KEY
   printf "\n"
   export NVIDIA_API_KEY
   ```

Keep the key private. Do not paste it into source files, `config.yaml`, screenshots, or Git commits. The experiment runner reads `NVIDIA_API_KEY` from the environment.

## Project structure

```text
data/                         Synthetic task sets and separate labels
docs/                         Article, figures, and Matplotlib generator
legacy_plumbing/              Compatibility code for older experiments
scripts/
  analysis/                   Result analysis and model comparisons
  audit/                      Saved-proposal audits and offline checks
  experiments/                Hosted and A3 smoke experiment runners
  legacy/                     Older experiment entry points
src/
  agents/                     Actor, Reflector, Curator, and API client
  analysis/                   Grounding checks and metrics
  loop/                       Learning loop and playbook provenance
  obs/                        Phoenix tracing and evaluation helpers
  testbed/                    Synthetic API, task generation, and grader
tests/                        Offline unit tests
config.yaml                   Model, rate limit, and reference-cost settings
```

## View the article

From the repository root, start a local web server:

```bash
python3 -m http.server 8000 --directory docs
```

Open [http://localhost:8000/learning-from-experience.html](http://localhost:8000/learning-from-experience.html). The server keeps the article’s figures available alongside the HTML page.
