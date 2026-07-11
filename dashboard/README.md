# TalkTuner dashboard (community frontend)

The TalkTuner paper describes a chat interface that shows — live — what the
chatbot's internal representation says about *you* (age, gender, education,
socioeconomic status), and lets you pin those beliefs to steer the model. The
original React/Flask app was never open-sourced; this directory is a
self-contained rebuild of it on top of the repo's probing method.

A hosted version runs at [talktuner.mcgee.cat](https://talktuner.mcgee.cat)
(Qwen2.5-7B-Instruct on a Hugging Face GPU Space — see [`deploy/`](../deploy/)).
The instructions below run it locally with a smaller model.

Because the paper's bundled probe checkpoints only work with
Llama-2-13b-chat's activations, this dashboard trains its own probes for a
smaller model (default: Llama-3.2-3B-Instruct, ~6.5 GB, runs well on Apple
Silicon via MPS) using the labeled conversations already shipped in
`data/dataset/`.

## Quickstart

```bash
# 1. Environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate flask scikit-learn numpy tqdm

# 2. Unzip the bundled training data
cd data/dataset && for z in *.zip; do mkdir -p "${z%.zip}" && unzip -n -q "$z" -d "${z%.zip}"; done; cd ../..

# 3. Train probes for your model (one-time, ~20-40 min on an M-series Mac;
#    the model downloads automatically on first run)
python dashboard/train_probes.py

# 4. Run the dashboard
python dashboard/server.py
# open http://127.0.0.1:5170
```

## How it works

- **Reading.** After every message, the server runs one forward pass over the
  conversation and takes the hidden state of the last token at every layer.
  Per attribute, a logistic-regression probe (trained by `train_probes.py` on
  the bundled labeled conversations, best layer selected by validation
  accuracy) turns that vector into class probabilities — the confidence bars.
- **Control.** Pinning a class adds the probe's (normalized) weight direction
  for that class to the residual stream at the layers just below the probe's
  layer during generation, scaled by the "steering strength" slider — the
  same intervention idea as the paper's controlling probes.

## Options

```bash
python dashboard/train_probes.py --model <hf-model> --per-class 300
python dashboard/server.py --model <hf-model> --port 5170
```

Probes are stored under `data/probe_checkpoints/<model-name>/` as small
`.npz` files with a `meta.json` (chosen layer + validation accuracy per
attribute). Train once per model; the server picks them up by model name.

## Caveats

- Probe accuracy varies by attribute (shown in each card's header); these are
  research probes on synthetic conversations, not reliable demographic
  inference. That's part of the point of the paper — look, and stay skeptical.
- One chat at a time: generation holds a lock, this is a local toy, not a
  deployment.
