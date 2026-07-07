"""Convert the paper's bundled Llama-2-13b-chat probe checkpoints into the
dashboard's probe format ("paper mode").

The repo ships reading and controlling probes for every layer (0-40) of
Llama-2-13b-chat. This script:
  1. extracts the linear weights from those checkpoints,
  2. picks the best reading layer per attribute by evaluating on a held-out
     sample of the bundled labeled conversations (requires running the 13B
     model, ~10 min on an M-series Mac),
  3. saves dashboard-format probes to data/probe_checkpoints/llama-2-13b-chat-hf/
     including the paper's steering setup (controlling probes for decoder
     layers 19-28, applied at strength N to the last token).

Reading probes were trained on llama_v2-formatted conversations with the last
assistant reply removed and an elicitation suffix appended (" I think the
<attribute> of this user is"); the dashboard reproduces that at serve time.

Usage:
    python dashboard/convert_paper_probes.py [--model meta-llama/Llama-2-13b-chat-hf]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common import ATTRIBUTES, REPO_ROOT, ProbeSet, iter_conversations, pick_device

sys.path.insert(0, str(REPO_ROOT))
from src.dataset import llama_v2_prompt  # noqa: E402

# Class order the paper's probes were trained with (train_read_and_
# controlling_probes.ipynb); age/socioeco/education match ATTRIBUTES,
# gender is male=0, female=1.
PAPER_CLASSES = dict(ATTRIBUTES, gender=["male", "female"])

ELICIT_SUFFIX = {
    "age": " I think the age of this user is",
    "gender": " I think the gender of this user is",
    "socioeco": " I think the socioeconomic status of this user is",
    "education": " I think the education level of this user is",
}

# Paper hyperparameters (causality notebooks): edit decoder layers 19-28
# using the controlling probe of layer i+1, strength N=7, last token only.
STEER_FROM, STEER_TO, STEER_N = 19, 29, 7

N_LAYERS = 41  # hidden-state indices 0..40 for the 13B


def load_probe_weights(kind, attr, layer):
    path = (REPO_ROOT / "data" / "probe_checkpoints" / f"{kind}_probe" /
            f"{attr}_probe_at_layer_{layer}.pth")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd["proj.0.weight"].numpy(), sd["proj.0.bias"].numpy()


def paper_text(messages, attr):
    """Reproduce the probe training text: llama_v2 format, last assistant
    reply removed, leading <s> stripped, elicitation suffix appended."""
    msgs = list(messages)
    if msgs and msgs[-1]["role"] == "assistant":
        msgs = msgs[:-1]
    text = llama_v2_prompt(msgs)
    if text.startswith("<s>"):
        text = text[len("<s>"):]
    return text + ELICIT_SUFFIX[attr]


@torch.no_grad()
def extract(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=2048).input_ids.to(device)
    out = model(ids, output_hidden_states=True)
    return np.stack([h[0, -1].float().cpu().numpy() for h in out.hidden_states])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-2-13b-chat-hf")
    ap.add_argument("--per-class", type=int, default=25,
                    help="held-out conversations per class for layer selection")
    args = ap.parse_args()

    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    print(f"Loading {args.model} on {device} (this is the 26GB one)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.to(device).eval()

    out_dir = REPO_ROOT / "data" / "probe_checkpoints" / args.model.split("/")[-1].lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"model": args.model, "mode": "paper", "attributes": {},
            "reading": {"strip_last_assistant": True, "suffix": ELICIT_SUFFIX},
            "steering": {"from": STEER_FROM, "to": STEER_TO, "n": STEER_N}}

    for attr, classes in PAPER_CLASSES.items():
        by_class = defaultdict(list)
        for messages, label in iter_conversations(attr):
            by_class[label].append(messages)
        rng = random.Random(1)  # different seed than any probe training
        sample = []
        for label, convs in sorted(by_class.items()):
            rng.shuffle(convs)
            sample += [(m, label) for m in convs[:args.per_class]]

        feats, labels = [], []
        for messages, label in tqdm(sample, desc=f"eval {attr}"):
            feats.append(extract(model, tokenizer, paper_text(messages, attr), device))
            labels.append(classes.index(label))
        X, y = np.stack(feats), np.array(labels)

        best = None
        for layer in range(N_LAYERS):
            coef, intercept = load_probe_weights("reading", attr, layer)
            pred = (X[:, layer] @ coef.T + intercept).argmax(1)
            acc = float((pred == y).mean())
            if best is None or acc > best[1]:
                best = (layer, acc)
        layer, acc = best
        print(f"[{attr}] best reading layer {layer}: acc {acc:.3f} on {len(y)} held-out")

        coef, intercept = load_probe_weights("reading", attr, layer)
        np.savez(out_dir / f"{attr}.npz", layer=layer, val_acc=acc,
                 classes=np.array(classes), coef=coef, intercept=intercept)

        # Controlling probes for the steering window: decoder layer i uses
        # the probe of hidden-state layer i+1.
        rows = np.stack([load_probe_weights("controlling", attr, i + 1)[0]
                         for i in range(STEER_FROM, STEER_TO)])
        np.savez(out_dir / f"{attr}_steering.npz", rows=rows)
        meta["attributes"][attr] = {"layer": layer, "val_acc": round(acc, 4)}

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved paper-mode probes to {out_dir}")
    ProbeSet(out_dir)  # reload smoke test
    print("Reload check OK")


if __name__ == "__main__":
    main()
