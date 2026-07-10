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

from common import (DEFAULT_MODEL, REPO_ROOT, SELF_REPORT, SYSTEM_PROMPT,
                    ProbeSet, pick_device, reading_hidden_states,
                    self_report_readings)

app = Flask(__name__, static_folder=None)
STATIC_DIR = Path(__file__).parent / "static"

# Guardrails for public deployments (e.g. a HF Space): bound the work one
# request can demand and fail politely when the single GPU is busy.
MAX_TURNS = 40
MAX_MSG_CHARS = 4000
BUSY_TIMEOUT_S = 90

state = {}
gen_lock = threading.Lock()


def load(model_name, probe_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # device_map streams shards straight to the accelerator; loading to CPU
    # first needs the full fp16 weights in system RAM (OOM on small hosts).
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16,
        device_map=device if device != "cpu" else None)
    model.eval()
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

    Generic mode: the controlling probe's unit direction scaled by alpha% of
    the training-time activation norm at each edited layer, applied at the 4
    decoder layers below the controlling probe's layer — norm calibration
    makes one alpha work across models with very different activation
    magnitudes (Qwen's norms dwarf Llama's). Paper mode: the controlling
    probes' raw weight rows * alpha at decoder layers 19-28, exactly as in
    the causality notebooks (N=7)."""
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
            direction, probe_layer, norms = probes.steering_direction(attr, target)
            top = min(probe_layer - 1, len(layers) - 1)
            for i in range(max(0, top - 3), top + 1):
                scale = (alpha / 100.0) * (norms[i + 1] if norms is not None
                                           else 100.0)
                per_layer.setdefault(i, []).append(
                    torch.tensor(direction * scale, dtype=torch.float16,
                                 device=device))

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
    """Probe readings for the conversation: one pass per attribute over the
    probes' training text (conversation with trailing assistant reply
    stripped + that attribute's elicitation suffix)."""
    model, tokenizer, device, probes = (state["model"], state["tokenizer"],
                                        state["device"], state["probes"])
    readings = {}
    for attr in probes.probes:
        if probes.mode == "paper":
            text = probes.paper_reading_text(messages, attr)
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=2048).input_ids.to(device)
            with torch.no_grad():
                out = model(ids, output_hidden_states=True)
            hs = np.stack([h[0, -1].float().cpu().numpy()
                           for h in out.hidden_states])
        else:
            hs = reading_hidden_states(
                model, tokenizer, messages,
                probes.meta["reading"]["suffix"][attr], device)
        readings[attr] = probes.read_attr(attr, hs)
    return readings


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
        # Attributes read by asking the model directly (read-only, no
        # steering). Skipped in paper mode: 4 extra 13B passes per turn is
        # too slow to be worth it.
        "self_report": ({attr: spec["classes"]
                         for attr, spec in SELF_REPORT.items()}
                        if probes.mode != "paper" else {}),
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
    if len(messages) > MAX_TURNS or any(
            len(m.get("content", "")) > MAX_MSG_CHARS for m in messages):
        return jsonify({"error": "conversation too long"}), 400

    def stream():
        from transformers import TextIteratorStreamer

        if not gen_lock.acquire(timeout=BUSY_TIMEOUT_S):
            yield sse({"type": "error",
                       "message": "The model is busy with another chat — "
                                  "try again in a minute."})
            return
        try:
            handles = steering_hooks(pins, alpha)
            try:
                # Reading 1: what the model believes right after your message.
                before = read_user_model(messages)
                if state["probes"].mode != "paper":
                    before.update(self_report_readings(
                        state["model"], state["tokenizer"], messages,
                        state["device"]))
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

                # Readings always elicit at the last user message (trailing
                # assistant replies are stripped, as in the paper), so the
                # post-reply reading equals the pre-reply one — reuse it.
                yield sse({"type": "done", "reply": reply,
                           "readings": before})
            finally:
                for h in handles:
                    h.remove()
        finally:
            gen_lock.release()

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
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 when serving from a container")
    args = ap.parse_args()
    load(args.model, args.probes)
    app.run(host=args.host, port=args.port, threaded=True)
