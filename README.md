# ha-semif-fastpath

A fast path for Home Assistant voice commands. A small model settles simple device commands
("turn off the desk lamp", "set the thermostat to 68") in under half a second, without running the
LLM. Anything it is unsure about goes to the LLM exactly as before.

It sits in front of Ollama as a proxy, so Home Assistant's Ollama integration needs no changes
beyond its URL.

| | |
|---|---|
| "turn off &lt;a light&gt;", end to end through HA | **0.58 s** (was 5–10 s through a local 27B LLM) |
| intent accuracy at p ≥ 0.8 | **97 %**, covering 94 % of commands |
| VRAM | **2.7 GB**, shares a card with the LLM |

## How a voice turn flows

```
satellite → Whisper → HA Assist (Ollama agent) ──/api/chat──► semif_proxy ──► Ollama (LLM)
                                                                  │   ▲
                                                          /decide ▼   │ unsure / not eligible / error
                                                              semif_service
```

1. HA's Ollama agent sends `/api/chat` to **`semif_proxy.py`** with its system prompt (your exposed
   entities) and the `Hass*` tools. The proxy keeps a copy of that prompt; semif reads its device
   list from it.
2. **Is this a fresh, single command?** Not if it answers a question the assistant just asked, not
   if it contains *and* / *then* / *also*. Otherwise → LLM.
3. **`semif_service.py`** decides: one forward pass picks the action from eight described options,
   then the device from the exposed entities whose domain fits (14 per pass, then a final round).
   Confidence is the **margin** between the top two devices. Unsure → LLM.
4. The proxy answers **as if it were the LLM**, with the tool call (`HassTurnOff {"name": "Desk Lamp"}`).
   HA executes it.
5. HA then asks the model to confirm. If exactly one device action succeeded, the proxy replies
   with **a single space**: HA skips TTS on a blank reply and the satellite's done chime is the
   confirmation. This applies to LLM-chosen calls too, so every device command saves a round trip.

Every rule fails open: semif down, slow (> 1.5 s), unsure, or any error means the LLM handles the
turn. The worst case is the old latency, never a wrong action.

## How semif decides

[SemIf](https://github.com/TheoLeeCJ/SemIf) (MIT) prompts an unmodified model with a state, a
question and labelled option descriptions, runs a single forward pass, and softmaxes the logits of
the option-label tokens. No sampling, no JSON parsing, and the answer comes with a real probability.
This uses Qwen3.5-4B (Q4_K_M GGUF) on llama.cpp; a decision is 2–4 passes, ~100–450 ms.

```
POST /decide {"text": "turn off the desk lamp"}
→ {"handled": true, "tool": {"name": "HassTurnOff", "arguments": {"name": "Desk Lamp"}},
   "action": "turn_off", "action_p": 1.0, "entity": "Desk Lamp", "entity_p": 0.86, "margin": 0.71, "latency_ms": 341}

POST /decide {"text": "lights off"}
→ {"handled": false, "entity_p": 0.41, "margin": 0.14, "why": "entity p 0.41 / margin 0.14 below 0.5/0.6", ...}
```

## Rules learned in production

- **Never offer an "ambiguous" option.** Given a "none / unclear" choice, the model took it whenever
  the name wasn't said verbatim (coverage fell to 50 %). Asked only "which one?", it was right 94 %
  of the time, and ambiguity shows up as a low margin instead.
- **Don't answer the answer.** "Turn off the lamp" → "which lamp?" → "the one by the desk". semif
  sees only the last line, has no verb, and once emitted *turn on*. Follow-ups go to the LLM.
- **One decision per command.** "Turn off X and Y": semif picked one device, the blank reply said
  nothing, the other stayed on. Compound commands go to the LLM.
- **A space, not an empty string.** HA drops an empty assistant message, retries up to its limit,
  then says "Unable to get response".
- **Keep a model name the proxy never touches.** If you regression-test your voice prompt, the
  tests must grade the LLM, not semif: use `INTERCEPT_MODELS`, or send `X-Semif: off`.
- **One exposed target per name.** Two exposed entities with the same name (a bulb and its Hue
  room) can't be addressed by name at all; HA returns `DUPLICATE_NAME`.
- **GPU, or don't bother.** SemIf's llama.cpp backend defaults to CPU (~1.8 s per pass);
  `start.sh` patches it to offload (~61 ms per pass on an RTX 3090).

## Measured (RTX 3090, small test sets from one house: direction, not a benchmark)

| model / backend | per pass | intent, prompt v1 | intent, prompt v2 | at p ≥ 0.8 (coverage) | VRAM |
|---|---|---|---|---|---|
| Qwen3-0.6B, torch, GPU | 14 ms | 30 % | — | — | ~1.5 GB |
| Qwen3.5-4B Q4_K_M, llama.cpp, CPU | 1,826 ms | 83 % | — | 87 % (91 %) | 0 |
| **Qwen3.5-4B Q4_K_M, llama.cpp, GPU** | **61 ms** | 83 % | **93 %** | **97 % (94 %)** | **2.7 GB** |

| device linker (31 specific + 5 ambiguous) | top-1 | ambiguous margin | latency | VRAM |
|---|---|---|---|---|
| **Qwen3.5-4B via SemIf (used here)** | **94 %** | 0.14–0.56 | 269 ms | 2.7 GB |
| Qwen3-Reranker-0.6B, zero-shot | 90 % | 0.00–0.12 | 141 ms | 1.3 GB |
| nomic-embed-text, cosine | 84 % | 0.00–0.06 | 3 ms | already loaded |

## Run it

1. Move Ollama to another port (or host) and run the proxy where HA expects Ollama. Pure stdlib
   Python 3.10+:
   ```bash
   python3 semif_proxy.py --selftest
   OLLAMA_URL=http://localhost:11435 python3 semif_proxy.py
   ```
2. On a machine with an NVIDIA GPU and ~3.5 GB free, run `start.sh`. It installs SemIf and
   llama-cpp-python, downloads the GGUF and starts `semif_service.py` on `:8090`. Set
   `SURFACE_URL=http://<proxy>:11434/debug/ha-surface` (and `SEMIF_URL` on the proxy if it is not
   `localhost:8090`). Tested on the `vllm/vllm-openai:v0.29.0` image; any CUDA image with Python and
   `transformers` should do. Mount `/models` so restarts skip the downloads.
3. Make one voice request so the proxy captures HA's prompt, then `GET /refresh` on semif (or wait
   five minutes). `GET /health` shows how many entities it sees.
4. Optional: `HOUSE_NOTES` describes how your devices are named; `WAKE_WORDS` (e.g.
   `hey nabu,okay nabu`) strips a wake word Whisper transcribed.

All settings are environment variables, documented at the top of each file.

**Scope:** HA's built-in intents only — on/off, brightness, temperature, open/close/stop on covers.
Scripts, questions, timers and everything else go to the LLM unchanged. The proxy speaks the Ollama
API, so it is for HA's Ollama integration specifically. ⚠️ `/debug/ha-surface` returns your exposed
entity list; keep the proxy on your LAN.

## Status

A cleaned-up extract of a larger private gateway where this runs daily. The proxy is tested with
stub backends (`--selftest`), not as this exact bundle against a live house; treat it as a working
starting point. Issues and PRs welcome.

MIT licensed. SemIf is MIT, by TheoLeeCJ. Qwen3.5-4B GGUF by bartowski.
