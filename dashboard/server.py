"""TalkTuner dashboard server.

Serves a chat UI backed by a local chat model, and after every turn reads the
model's internal representation of the user (age, gender, education,
socioeconomic status) with the probes trained by train_probes.py. Pinning an
attribute steers the model's activations toward that class during generation.

Usage:
    python dashboard/server.py [--model MODEL] [--port 5170]
"""

import argparse
import json
import threading
from pathlib import Path

import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_from_directory

from common import (DEFAULT_MODEL, REPO_ROOT, SYSTEM_PROMPT, ProbeSet,
                    last_token_hidden_states, pick_device)

app = Flask(__name__, static_folder=None)
STATIC_DIR = Path(__file__).parent / "static"

state = {}
gen_lock = threading.Lock()


def load(model_name, probe_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model.to(device).eval()
    probe_dir = Path(probe_dir) if probe_dir else (
        REPO_ROOT / "data" / "probe_checkpoints" / model_name.split("/")[-1].lower())
    probes = ProbeSet(probe_dir)
    state.update(model=model, tokenizer=tokenizer, device=device,
                 probes=probes, model_name=model_name)
    print(f"Loaded {model_name} on {device}; probes from {probe_dir}")


def steering_hooks(pins, alpha):
    """Register forward hooks that push the last token's hidden state toward
    the pinned classes (the paper edits only the last position; during cached
    generation every new token is the last position, so it is steered too).

    Generic mode: normalized probe direction * alpha at the 4 decoder layers
    below the probe layer. Paper mode: the controlling probes' weight rows *
    alpha at decoder layers 19-28, as in the causality notebooks (N=7)."""
    model, probes, device = state["model"], state["probes"], state["device"]
    layers = model.model.layers
    per_layer = {}
    for attr, target in pins.items():
        if not target:
            continue
        if probes.mode == "paper":
            for i, row in probes.paper_steering_rows(attr, target).items():
                per_layer.setdefault(i, []).append(
                    torch.tensor(row * alpha, dtype=torch.float16, device=device))
        else:
            direction, probe_layer = probes.steering_direction(attr, target)
            vec = torch.tensor(direction * alpha, dtype=torch.float16, device=device)
            top = min(probe_layer - 1, len(layers) - 1)
            for i in range(max(0, top - 3), top + 1):
                per_layer.setdefault(i, []).append(vec)

    handles = []
    for i, vecs in per_layer.items():
        shift = torch.stack(vecs).sum(0)

        def hook(module, inputs, output, shift=shift):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            hidden[:, -1, :] += shift
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        handles.append(layers[i].register_forward_hook(hook))
    return handles


def read_user_model(messages):
    """Probe readings for the conversation. Paper mode replicates the probes'
    training text per attribute (llama_v2 format, last assistant reply
    stripped, elicitation suffix), so it runs one pass per attribute."""
    model, tokenizer, device, probes = (state["model"], state["tokenizer"],
                                        state["device"], state["probes"])
    if probes.mode == "paper":
        readings = {}
        for attr in probes.probes:
            text = probes.paper_reading_text(messages, attr)
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=2048).input_ids.to(device)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            hs = np.stack([h[0, -1].float().cpu().numpy()
                           for h in out.hidden_states])
            readings[attr] = probes.read_attr(attr, hs)
        return readings
    hs = last_token_hidden_states(model, tokenizer, messages, device)
    return probes.read(hs)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC_DIR, name)


@app.route("/api/config")
def config():
    probes = state["probes"]
    return jsonify({
        "model": state["model_name"],
        "mode": probes.mode,
        "default_alpha": probes.default_alpha(),
        "max_alpha": probes.max_alpha(),
        "attributes": {attr: {"classes": p["classes"], "layer": p["layer"],
                              "val_acc": p["val_acc"]}
                       for attr, p in probes.probes.items()},
    })


def generation_inputs(messages):
    """Tokenized prompt for generation. Paper mode uses the paper's own
    llama_v2 formatting (and its default system prompt) so that generation
    and probe reading see the same conversation format."""
    tokenizer, device = state["tokenizer"], state["device"]
    if state["probes"].mode == "paper":
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from src.dataset import llama_v2_prompt

        text = llama_v2_prompt(messages)
        if text.startswith("<s>"):
            text = text[len("<s>"):]
        return tokenizer(text, return_tensors="pt").to(device)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        add_generation_prompt=True, return_tensors="pt",
        return_dict=True).to(device)


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json()
    messages = body["messages"]
    pins = body.get("pins") or {}
    alpha = float(body.get("alpha", 8.0))

    def stream():
        from transformers import TextIteratorStreamer

        with gen_lock:
            handles = steering_hooks(pins, alpha)
            try:
                # Reading 1: what the model believes right after your message.
                before = read_user_model(messages)
                yield sse({"type": "readings", "when": "before",
                           "readings": before})

                tokenizer, model = state["tokenizer"], state["model"]
                enc = generation_inputs(messages)
                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True)
                thread = threading.Thread(target=model.generate, kwargs=dict(
                    **enc, streamer=streamer, max_new_tokens=512,
                    do_sample=True, temperature=0.7, top_p=0.9))
                thread.start()
                reply = ""
                for token in streamer:
                    reply += token
                    yield sse({"type": "token", "text": token})
                thread.join()

                # Reading 2: beliefs after the model's own reply. The paper's
                # probes always read at the last user message (the reply is
                # stripped), so there the reading is unchanged — reuse it.
                if state["probes"].mode == "paper":
                    after = before
                else:
                    full = messages + [{"role": "assistant", "content": reply}]
                    after = read_user_model(full)
                yield sse({"type": "done", "reply": reply, "readings": after})
            finally:
                for h in handles:
                    h.remove()

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--port", type=int, default=5170)
    args = ap.parse_args()
    load(args.model, args.probes)
    app.run(host="127.0.0.1", port=args.port, threaded=True)
