# PII Privacy Gate

Before a request goes to a model that is not on this machine, omp scans text it has not cleared yet for personal data (names, e-mail addresses, phone numbers, ID and card numbers, and similar) with a small local model, and asks what to do if it finds some. Disabled by default; this is a prompt-level safeguard, not a compliance guarantee.

## Enabling

Toggle `privacy.piiGate` in `/settings` (Providers → Privacy) or `config.yml`:

```yaml
privacy:
  piiGate: true
  localModel: ollama/gemma4:e4b # optional
```

`privacy.localModel` is a model selector in the same format as the model roles (`provider/id`). It is the model the gate offers to switch to; it must resolve to an available model that stays on this machine (see [Local models](#local-models)), otherwise it is not offered.

The detector is `onnx-community/bert-small-pii-detection-ONNX` (about 29 MB, downloaded on first use) running in omp's tiny worker, so scanned text never leaves the machine. It ignores categories that are noise in a coding session (dates, URLs, organizations, titles, nationality and group labels) and low-confidence spans.

## What is scanned

Text the next request carries that the gate has not cleared yet:

- your prompts
- tool results (file reads, command output, search hits, …)
- output of shell and Python commands you ran yourself, and `@file` attachments
- compaction and branch summaries

Assistant output, the system prompt, extension/custom messages, and images are not scanned. A text is scanned once per session: it is cleared when it scans clean or when you choose to send it, and it is not asked about again. A resumed session scans its history once more.

## When personal data is found

The request pauses and omp asks, listing what was found and where, for example `EMAIL_ADDRESS ×2 (d•••@meridiancap.com), PHONE_NUMBER ×1 (+•••45) in your prompt, read output`:

- **Send to `<provider/model>`** sends the request; that text is cleared for the rest of the session.
- **Switch to local model `<privacy.localModel>`** (only when it is set and resolves) switches the session to it for this session and sends the same request there. Nothing is cleared, so switching back to a cloud model asks again.
- **Cancel** (or dismissing the dialog) sends nothing and ends the turn. There is no retry, no fallback model, and no provider error; the turn shows the gate's message. The message stays in the session, so a later request to a cloud model asks again until you switch to a local model or start a new session.

When `privacy.localModel` is unset, the question also suggests a local model sized to this machine (Apple Silicon unified memory, an NVIDIA GPU's VRAM, or system RAM) with the `ollama pull` command to fetch it.

## Without a prompt UI

Print mode (`-p`), RPC and ACP hosts whose client cannot show prompts, and subagents have nobody to ask, so the gate fails closed: the request is not sent and the turn ends with a message naming what was found. Turn `privacy.piiGate` off, or run with a local model. The same happens when the detector cannot run (for example the model could not be downloaded): the gate never sends text it could not scan.

## Local models

A model counts as local when it runs in-process or its base URL is loopback, a private network address (RFC 1918), or a `.local` host. That includes a model server elsewhere on your LAN. Requests to a local model are not scanned.

## Session titles

Automatic titles and titles `/rename` generates go to the title model directly, so the gate only lets them use text it has cleared or that scans clean. A prompt with uncleared personal data leaves the session untitled.

## Known gaps

The gate sits on requests built from session messages. Features that call a provider directly with their own input bypass it:

- LLM judges, commit message generation, memory (mnemopi) extraction, auto-repair, `annotate`, advisors, auto-learn capture, and subagent task labels.
- Side requests built from session messages (`/btw`, handoff, compaction, the title fork) are gated against the session's active model, which may differ from the model they call, and can only be sent or cancelled, not switched.
- Subagents run their own gate with an empty cleared set and no prompt UI, so personal data in a subagent's context fails closed even if you approved it in the parent session.
- Detection is statistical: it misses some personal data and flags some code. Image content is not read.

## Key files

- `packages/coding-agent/src/privacy/pii-gate.ts` -- the gate: scan, cleared set, prompt, switch/cancel
- `packages/coding-agent/src/privacy/pii-detector.ts` -- local detector and its label/score policy
- `packages/coding-agent/src/privacy/local-model-advice.ts` -- hardware probe and local model suggestion
- `packages/coding-agent/src/privacy/settings.ts` -- `privacy.piiGate` and `privacy.localModel`
- `packages/coding-agent/src/sdk.ts` -- wires the gate into the session's `transformContext`
