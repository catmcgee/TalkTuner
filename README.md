# TalkTuner — a live dashboard for what an LLM thinks about you

Chat with a language model and watch, live, what its internal representation
says about *you* — age, gender, education, socioeconomic status — then pin a
belief to override what it thinks and see its answers change.

**Try it: [talktuner.mcgee.cat](https://talktuner.mcgee.cat)** (the first
message may take a minute — the GPU Space wakes from sleep).

This is a fork of
[yc015/TalkTuner-chatbot-llm-dashboard](https://github.com/yc015/TalkTuner-chatbot-llm-dashboard),
the research code for the paper
["Designing a Dashboard for Transparency and Control of Conversational AI"](https://arxiv.org/abs/2406.07882)
(Chen et al.). The paper demos a chat UI with this dashboard, but that app was
never open-sourced — the original repo contains the probing method and
datasets only. This fork builds and deploys the missing interface.

## What this fork adds

Everything below is new in this fork; none of it exists upstream.

- **[`dashboard/`](dashboard/) — the chat UI and server.** A self-contained
  rebuild of the paper's interface: a Flask server that runs the model with
  `output_hidden_states`, reads each attribute with a logistic-regression
  probe after every message, and steers generation by adding probe directions
  to the residual stream when you pin a belief. Vanilla JS frontend, no build
  step. See [dashboard/README.md](dashboard/README.md) for a local
  quickstart and how reading/control work.
- **[`dashboard/train_probes.py`](dashboard/train_probes.py) — probes for any
  model.** The paper's released probe checkpoints only fit
  Llama-2-13b-chat's activations. This script retrains reading and steering
  probes for any Hugging Face model from the labeled conversations bundled in
  `data/dataset/`, picking the best layer per attribute by validation
  accuracy (steering probes constrained to mid-network layers, where
  interventions actually work).
- **[`deploy/`](deploy/) — the hosted version.** A Docker Hugging Face Space
  ([catmcgee/talktuner](https://huggingface.co/spaces/catmcgee/talktuner))
  runs the server on GPU with Qwen2.5-7B-Instruct and probes trained remotely
  on HF Jobs; Vercel serves the static UI at
  [talktuner.mcgee.cat](https://talktuner.mcgee.cat) and proxies `/api/*` to
  the Space ([vercel.json](vercel.json)).
- Trained probe checkpoints for Llama-3.2-3B-Instruct (the local default —
  small enough to run on a MacBook) and Qwen2.5-7B-Instruct (what the hosted
  demo runs) under `data/probe_checkpoints/`.

A few honest caveats: the probes are research artifacts trained on synthetic
conversations, not reliable demographic inference — watching them be
confidently wrong is part of the point. The hosted demo serves one
conversation at a time, and nothing you type is stored.

## What's from the original repo

The probing method and experiments from the paper, unchanged:

- [`src/`](src/) — probe training/intervention code used in the paper's
  experiments
- [`notebooks/`](notebooks/) — the paper's analysis notebooks
- [`data/dataset/`](data/) — the labeled synthetic conversations the probes
  are trained on, plus the original Llama-2-13b probe checkpoints
- [`environment.yml`](environment.yml) — the paper's conda environment
  (`conda env create -f environment.yml && conda activate talktuner-gpu`)

For the paper's video demo and details, see the authors' project page:
[yc015.github.io/TalkTuner-a-dashboard-ui-for-chatbot-llm](https://yc015.github.io/TalkTuner-a-dashboard-ui-for-chatbot-llm/).

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate flask scikit-learn numpy tqdm

# unzip the bundled training data
cd data/dataset && for z in *.zip; do mkdir -p "${z%.zip}" && unzip -n -q "$z" -d "${z%.zip}"; done; cd ../..

python dashboard/train_probes.py   # one-time, ~20-40 min on an M-series Mac
python dashboard/server.py         # open http://127.0.0.1:5170
```

Full details, options, and how it works: [dashboard/README.md](dashboard/README.md).

## Citation

If you use the probing method or datasets, cite the original paper:

> Yida Chen, Aoyu Wu, Trevor DePodesta, Catherine Yeh, Kenneth Li, Nicholas
> Castillo Marin, Oam Patel, Jan Riecke, Shivam Raval, Olivia Seow, Martin
> Wattenberg, Fernanda Viégas. *Designing a Dashboard for Transparency and
> Control of Conversational AI.* [arXiv:2406.07882](https://arxiv.org/abs/2406.07882)
