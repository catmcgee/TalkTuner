"""Shared pieces for the TalkTuner dashboard: dataset parsing, activation
extraction, and the linear probe format used by both training and serving.

The bundled probe checkpoints in data/probe_checkpoints/ were trained on
Llama-2-13b-chat and do not transfer to other models, so the dashboard trains
its own probes (see train_probes.py) on whichever chat model it is configured
to run.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "dataset"

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."

# Display order matters: the UI renders classes in this order.
ATTRIBUTES = {
    "age": ["child", "adolescent", "adult", "older adult"],
    "gender": ["female", "male"],
    "education": ["someschool", "highschool", "collegemore"],
    "socioeco": ["low", "middle", "high"],
}

# Elicitation suffixes for probe reading, from the paper's training pipeline
# (dataset.py appends this after the last user message; the probe reads the
# suffix's final token). Reading on the bare conversation instead makes the
# probe over-weight writing style: "lol im 28 btw" reads as adolescent. The
# suffix makes the model aggregate its evidence — including stated facts.
READING_SUFFIX = {
    "age": " I think the age of this user is",
    "gender": " I think the gender of this user is",
    "socioeco": " I think the socioeconomic status of this user is",
    "education": " I think the education level of this user is",
}

# Attributes read by asking the model directly (next-token probabilities over
# the candidate words after an elicitation suffix) rather than with trained
# probes. Read-only in the UI: no probe weights means nothing to steer with.
# Candidate words must differ in their first token.
SELF_REPORT = {
    "mood": {
        "suffix": " I think the current mood of this user is",
        "classes": ["happy", "neutral", "stressed", "sad"],
    },
    "tech expertise": {
        "suffix": " I think the technical expertise of this user is",
        "classes": ["beginner", "intermediate", "advanced"],
    },
    "english fluency": {
        "suffix": " I think this user's English is",
        "classes": ["native", "fluent", "basic"],
    },
    "personality": {
        "suffix": " I think this user is more",
        "classes": ["introverted", "extroverted"],
    },
}

# Folder-name prefix on disk -> attribute key above.
FOLDER_ATTR = {
    "age": "age",
    "gender": "gender",
    "education_three_classes": "education",
    "socioeconomic": "socioeco",
}

_FILE_RE = re.compile(r"conversation_\d+_(?:[a-z]+)_([a-z ]+)\.txt$")


def iter_conversations(attribute):
    """Yield (messages, class_label) for every bundled conversation of an
    attribute. The class label comes from the filename's last segment; the
    attribute comes from the folder name (some gender files are misnamed
    *_age_male.txt, so the filename attribute segment is not trusted)."""
    for folder in sorted(DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue
        m = re.match(r"(?:llama|openai)_(.+?)_\d+$", folder.name)
        if not m or FOLDER_ATTR.get(m.group(1)) != attribute:
            continue
        for path in sorted(folder.glob("*.txt")):
            fm = _FILE_RE.search(path.name)
            if not fm:
                continue
            label = fm.group(1)
            if label not in ATTRIBUTES[attribute]:
                continue
            messages = parse_conversation(path.read_text(encoding="utf-8"))
            if len(messages) >= 2:
                yield messages, label


_SPEAKER_RE = re.compile(
    r"^(?:###\s*)?(HUMAN|Human|ASSISTANT|Assistant):\s*(.*)$")


def parse_conversation(text):
    """Parse a transcript into chat messages. The llama-generated files use
    HUMAN:/ASSISTANT: prefixes; the openai-generated ones use ### Human: /
    ### Assistant:."""
    messages = []
    role = None
    buf = []
    for line in text.splitlines():
        m = _SPEAKER_RE.match(line)
        if m:
            _flush(messages, role, buf)
            role = "user" if m.group(1).lower() == "human" else "assistant"
            buf = [m.group(2).strip()]
        elif role is not None:
            buf.append(line)
    _flush(messages, role, buf)
    return messages


def _flush(messages, role, buf):
    if role is not None:
        content = "\n".join(buf).strip()
        if content:
            messages.append({"role": role, "content": content})


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_chat(tokenizer, messages):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def reading_text(tokenizer, messages, suffix):
    """Conversation formatted for an elicitation reading: trailing assistant
    reply stripped (the paper's remove_last_ai_response), generation prompt
    added, suffix started as the assistant's answer."""
    msgs = list(messages)
    if msgs and msgs[-1]["role"] == "assistant":
        msgs = msgs[:-1]
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}] + msgs,
        tokenize=False, add_generation_prompt=True)
    return text + suffix


@torch.no_grad()
def reading_hidden_states(model, tokenizer, messages, suffix, device,
                          max_tokens=2048):
    """Hidden states of the elicitation suffix's last token at every layer:
    array [n_layers+1, dim]."""
    text = reading_text(tokenizer, messages, suffix)
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens, add_special_tokens=False
                    ).input_ids.to(device)
    out = model(ids, output_hidden_states=True)
    return np.stack([h[0, -1].float().cpu().numpy() for h in out.hidden_states])


@torch.no_grad()
def self_report_readings(model, tokenizer, messages, device, max_tokens=2048):
    """{attr: {class: prob}} by asking the model to complete an elicitation
    suffix and comparing next-token probabilities of the candidate words."""
    readings = {}
    for attr, spec in SELF_REPORT.items():
        text = reading_text(tokenizer, messages, spec["suffix"])
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens, add_special_tokens=False
                        ).input_ids.to(device)
        logits = model(ids).logits[0, -1].float()
        cand = [tokenizer(f" {c}", add_special_tokens=False).input_ids[0]
                for c in spec["classes"]]
        probs = torch.softmax(logits[cand], dim=0).cpu().numpy()
        readings[attr] = {c: float(p) for c, p in zip(spec["classes"], probs)}
    return readings


class ProbeSet:
    """All trained probes for one model: probes[attr] = dict with keys
    layer (int), classes (list), coef [C, dim], intercept [C], val_acc.

    Two modes:
    - "generic" (default): probes from train_probes.py; read on the plain
      chat-templated conversation; steer with the normalized probe direction
      at the layers below the probe layer.
    - "paper": the original TalkTuner checkpoints converted by
      convert_paper_probes.py; read on the llama_v2-formatted conversation
      with the paper's elicitation suffix; steer with the controlling probes
      at decoder layers 19-28 exactly as in the causality notebooks."""

    def __init__(self, probe_dir):
        self.probe_dir = Path(probe_dir)
        with open(self.probe_dir / "meta.json") as f:
            self.meta = json.load(f)
        self.mode = self.meta.get("mode", "generic")
        self.probes = {}
        for attr in self.meta["attributes"]:
            data = np.load(self.probe_dir / f"{attr}.npz")
            self.probes[attr] = {
                "layer": int(data["layer"]),
                "classes": list(data["classes"]),
                "coef": data["coef"],
                "intercept": data["intercept"],
                "val_acc": float(data["val_acc"]),
            }
            steer = self.probe_dir / f"{attr}_steering.npz"
            if steer.exists():
                self.probes[attr]["steering_rows"] = np.load(steer)["rows"]

    def default_alpha(self):
        # Paper mode uses the notebooks' N=7; the generic normalized-direction
        # steering saturates a small model much sooner (tuned on the 3B).
        return self.meta.get("steering", {}).get("n", 3)

    def max_alpha(self):
        return self.meta.get("steering", {}).get("n", 3) * 2 + 2

    def read_attr(self, attr, hidden_states):
        p = self.probes[attr]
        h = hidden_states[p["layer"]]
        logits = p["coef"] @ h + p["intercept"]
        if len(logits) == 1:  # sklearn binary: single logit for class[1]
            pos = 1.0 / (1.0 + np.exp(-logits[0]))
            probs = np.array([1.0 - pos, pos])
        elif self.mode == "paper":  # trained with per-class sigmoid
            s = 1.0 / (1.0 + np.exp(-logits))
            probs = s / s.sum()
        else:
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
        return {c: float(pr) for c, pr in zip(p["classes"], probs)}

    def read(self, hidden_states):
        """hidden_states: [n_layers+1, dim] from last_token_hidden_states.
        Returns {attr: {class: probability}}."""
        return {attr: self.read_attr(attr, hidden_states) for attr in self.probes}

    def paper_reading_text(self, messages, attr):
        """The exact text the paper's reading probes were trained on."""
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from src.dataset import llama_v2_prompt

        msgs = [m for m in messages]
        if msgs and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        text = llama_v2_prompt(msgs)
        if text.startswith("<s>"):
            text = text[len("<s>"):]
        return text + self.meta["reading"]["suffix"][attr]

    def steering_direction(self, attr, target_class):
        """Generic mode: unit vector toward target_class + the probe layer."""
        p = self.probes[attr]
        coef, classes = p["coef"], p["classes"]
        idx = classes.index(target_class)
        if len(coef) == 1:  # binary probe: coef points toward classes[1]
            w = coef[0] if idx == 1 else -coef[0]
        else:
            others = np.mean([coef[i] for i in range(len(coef)) if i != idx], axis=0)
            w = coef[idx] - others
        return (w / np.linalg.norm(w)).astype(np.float32), p["layer"]

    def paper_steering_rows(self, attr, target_class):
        """Paper mode: {decoder_layer: weight row} for the steering window."""
        p = self.probes[attr]
        idx = p["classes"].index(target_class)
        start = self.meta["steering"]["from"]
        return {start + j: p["steering_rows"][j][idx].astype(np.float32)
                for j in range(len(p["steering_rows"]))}
