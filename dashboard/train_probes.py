"""Train TalkTuner-style reading probes for a given chat model.

For each attribute (age, gender, education, socioeco) this samples the
bundled labeled conversations, extracts the model's last-token hidden state
at every layer, trains a logistic-regression probe per layer, and saves the
best-performing layer's probe.

Usage:
    python dashboard/train_probes.py [--model MODEL] [--per-class N]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import (ATTRIBUTES, DEFAULT_MODEL, READING_SUFFIX, REPO_ROOT,
                    ProbeSet, iter_conversations, pick_device,
                    reading_hidden_states)


def sample_conversations(attribute, per_class, seed=0):
    by_class = defaultdict(list)
    for messages, label in iter_conversations(attribute):
        by_class[label].append(messages)
    rng = random.Random(seed)
    sampled = []
    for label, convs in sorted(by_class.items()):
        rng.shuffle(convs)
        for messages in convs[:per_class]:
            sampled.append((messages, label))
    rng.shuffle(sampled)
    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--out", default=None,
                    help="output dir (default: data/probe_checkpoints/<model name>)")
    args = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.to(device).eval()

    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "data" / "probe_checkpoints" / args.model.split("/")[-1].lower())
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"model": args.model, "attributes": {}, "per_class": args.per_class,
            "reading": {"suffix": READING_SUFFIX, "strip_last_assistant": True}}
    for attr, classes in ATTRIBUTES.items():
        data = sample_conversations(attr, args.per_class)
        print(f"\n[{attr}] {len(data)} conversations ({len(classes)} classes)")
        feats, bare_feats, labels = [], [], []
        for messages, label in tqdm(data, desc=f"extract {attr}"):
            suffix_states, bare_states = reading_hidden_states(
                model, tokenizer, messages, READING_SUFFIX[attr], device,
                with_bare=True)
            feats.append(suffix_states)
            bare_feats.append(bare_states)
            labels.append(label)
        y = np.array(labels)
        idx_train, idx_val = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=0, stratify=y)

        # Reading probes on the elicitation suffix (aggregates evidence like
        # stated facts); controlling probes on the bare conversation (their
        # directions steer generation, as in the paper).
        results = {}
        for kind, X in [("reading", np.stack(feats)),
                        ("controlling", np.stack(bare_feats))]:
            best = None
            for layer in range(X.shape[1]):
                clf = LogisticRegression(max_iter=2000, C=0.1)
                clf.fit(X[idx_train, layer], y[idx_train])
                acc = clf.score(X[idx_val, layer], y[idx_val])
                if best is None or acc > best[1]:
                    best = (layer, acc, clf)
            layer, acc, clf = best
            # Order coef rows to match ATTRIBUTES order for multiclass;
            # sklearn binary probes keep their single row (positive class =
            # classes_[1]).
            if len(clf.classes_) > 2:
                order = [list(clf.classes_).index(c) for c in classes]
                coef, intercept = clf.coef_[order], clf.intercept_[order]
                class_names = classes
            else:
                coef, intercept = clf.coef_, clf.intercept_
                class_names = list(clf.classes_)
            results[kind] = (layer, acc, coef, intercept, class_names)
            print(f"[{attr}] best {kind} layer {layer}: val acc {acc:.3f}")

        layer, acc, coef, intercept, class_names = results["reading"]
        np.savez(out_dir / f"{attr}.npz", layer=layer, val_acc=acc,
                 classes=np.array(class_names), coef=coef, intercept=intercept)
        c_layer, c_acc, c_coef, _, _ = results["controlling"]
        norms = np.linalg.norm(np.stack(bare_feats), axis=2).mean(axis=0)
        np.savez(out_dir / f"{attr}_steering.npz", layer=c_layer,
                 val_acc=c_acc, coef=c_coef, norms=norms)
        meta["attributes"][attr] = {
            "layer": layer, "val_acc": round(acc, 4),
            "steering_layer": c_layer, "steering_acc": round(c_acc, 4)}

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved probes to {out_dir}")
    print(json.dumps(meta["attributes"], indent=2))

    ProbeSet(out_dir)  # smoke test: reload everything
    print("Reload check OK")


if __name__ == "__main__":
    main()
