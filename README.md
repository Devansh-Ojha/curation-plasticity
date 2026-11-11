# Learning from Experience Without Learning from Noise

> **View the project article locally:** [Does Curating What an Agent Learns Raise Plasticity per Dollar?](docs/learning-from-experience.html)
>
> ```bash
> python3 -m http.server 8000 --directory docs
> ```
>
> Then open [http://localhost:8000/learning-from-experience.html](http://localhost:8000/learning-from-experience.html).

This repository is a small experiment on helping API agents learn reusable instructions from past attempts. The article covers the research background, design, saved results, Arize Phoenix workflow, and limitations.

## Setup

Use Python 3.10 or newer. From the repository root, install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-phoenix.txt
```

Hosted experiment runs require an NVIDIA API key. Sign in at the [NVIDIA API Catalog](https://build.nvidia.com/explore), choose a model marked **Free Endpoint**, and select **Get API Key**. See the [NVIDIA API Catalog quickstart](https://docs.api.nvidia.com/nim/docs/api-quickstart). Free access and rate limits depend on the model and current NVIDIA terms.

Set the key in your terminal without typing it into shell history:

```bash
printf "NVIDIA API key: "
read -s NVIDIA_API_KEY
printf "\n"
export NVIDIA_API_KEY
```

Keep the key private. Do not add it to source files or commit it to Git.
