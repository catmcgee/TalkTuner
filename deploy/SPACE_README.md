---
title: TalkTuner
emoji: 🪞
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Chat with an LLM and watch its internal model of you
---

# TalkTuner

Chat with a language model and watch — live — what its internal
representation says about *you*: age, gender, education, socioeconomic
status, and more. Pin a belief to override what the model thinks and see its
answers change.

A community rebuild of the interface from
["Designing a Dashboard for Transparency and Control of Conversational AI"](https://arxiv.org/abs/2406.07882)
(Chen et al.), running Llama-3.2-3B-Instruct with probes retrained on the
paper's published datasets. Source:
[catmcgee/TalkTuner](https://github.com/catmcgee/TalkTuner).

Notes:

- One conversation at a time (single GPU) — if it reports busy, wait a moment.
- The readings are research probes, not reliable demographic inference.
  Watching them be confidently wrong is part of the point.
- Nothing you type is stored; conversations live only in your browser tab.
