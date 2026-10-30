#!/usr/bin/env python3
"""semif_service — a fast decision layer for Home Assistant voice commands. One HTTP call,
~100-450 ms on a GPU, no text generated.

WHY. A device command routed through an LLM costs 5-10 s and produces a sentence nobody wants to
hear. This service answers one question per utterance — WHAT action, on WHICH exposed device, and
HOW SURE — by reading option probabilities off Qwen3.5-4B in single forward passes, using SemIf
(github.com/TheoLeeCJ/SemIf, MIT). The proxy (semif_proxy.py) turns a confident answer into the HA
tool call itself, so the LLM never runs; an unsure answer falls through to the LLM exactly as before.

HOW IT DECIDES
  1. ACTION: one forward pass over eight described options (turn_on, turn_off, set_brightness, …,
     not_a_device_command). Below ACTION_P → defer.
  2. DEVICE: only the exposed entities whose domain fits the action are offered, 14 at a time; the
     winners of each chunk play a final round. The answer is the top option; the CONFIDENCE is the
     MARGIN between the top two.
  3. ARGUMENTS: brightness / temperature are a regex on the text, never generated.

LESSONS (measured, not guessed)
  🔥 Never offer the model an "ambiguous" / "none of these" option for the device. It picks that
     whenever the name is not said verbatim (coverage fell to 50 %). Asked only "which one", it was
     right 94 % of the time, and ambiguity shows up as a LOW MARGIN instead ("lights off" with
     several lights exposed: margin 0.14-0.56; confident picks: >0.6).
  ⚠️ Action accuracy was 97 % at action_p >= 0.8, covering 94 % of commands.
  ⚠️ SemIf's llama.cpp backend hard-codes n_gpu_layers = 0 (CPU: ~1.8 s per pass). start.sh patches
     it to 99 (GPU: ~61 ms per pass, 2.7 GB VRAM for the Q4_K_M GGUF).
  ⚠️ Not thread-safe: one lock around every decision.

Entities come from the proxy's capture of HA's system prompt (GET /debug/ha-surface): HA's Ollama
integration lists every exposed entity there as "- names: …\n  domain: …\n  areas: …", so the
list is always what HA actually exposes. Refreshed every REFRESH_S seconds; a rename or unexpose
in HA needs no restart.

    POST /decide {"text": "turn off the reading lamp"}
    -> {"handled": true, "tool": {"name": "HassTurnOff", "arguments": {"name": "Reading Lamp"}},
        "action": "turn_off", "action_p": 1.0, "entity": "Reading Lamp", "entity_p": 0.86,
        "margin": 0.71, "latency_ms": 341}
    -> {"handled": false, "why": "entity p 0.41 / margin 0.14 below 0.5/0.6", ...}   # LLM takes it

    Optional per-request overrides: {"text": …, "thresholds": {"action_p", "margin", "entity_p"}}

Env: GGUF (path, set by start.sh), SURFACE_URL, PORT (8090), REFRESH_S (300), ACTION_P (0.8),
     MARGIN (0.6), ENTITY_P (0.5), THREADS (8), HOUSE_NOTES (free text the model sees with every
     request — naming conventions in your house), WARM_TEXT (a command run once at startup).
"""
import json, os, re, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SURFACE_URL = os.environ.get("SURFACE_URL", "http://localhost:11434/debug/ha-surface")
GGUF = os.environ["GGUF"]
PORT = int(os.environ.get("PORT", "8090"))
REFRESH_S = int(os.environ.get("REFRESH_S", "300"))
_ACTION_P = float(os.environ.get("ACTION_P", "0.8"))
_MARGIN = float(os.environ.get("MARGIN", "0.6"))
_ENTITY_P = float(os.environ.get("ENTITY_P", "0.5"))
DOMAINS = ("light", "switch", "climate", "cover")

from semif_phase1 import llamacpp_backend as B  # noqa: E402
MODEL, TOK, META = B.load_model("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", GGUF,
                                threads=int(os.environ.get("THREADS", "8")), context_tokens=4096)
LOCK = threading.Lock()

ACTIONS = [("turn_on", "Turn something on, start it, enable it, heat or warm something up."),
           ("turn_off", "Turn something off, kill it, shut it off, disable it."),
           ("set_brightness", "Dim or brighten a light, or set it to a level or percentage."),
           ("set_temperature", "Set a thermostat or climate to a temperature, make a room warmer or cooler."),
           ("open", "Open, raise or crack a door, window, cover or blind."),
           ("close", "Close, shut or lower a door, window, cover or blind."),
           ("stop", "Stop something that is moving or in progress."),
           ("not_a_device_command", "Not a command to a device at all: a question, a timer, small talk.")]
DOM = {"turn_on": ("light", "switch"), "turn_off": ("light", "switch"), "set_brightness": ("light",),
       "set_temperature": ("climate",), "open": ("cover",), "close": ("cover",), "stop": ("cover",)}
TOOL = {"turn_on": "HassTurnOn", "turn_off": "HassTurnOff", "set_brightness": "HassLightSet",
        "set_temperature": "HassClimateSetTemperature", "open": "HassTurnOn", "close": "HassTurnOff", "stop": "HassStopMoving"}
# What the model is told about your naming before every decision. Edit for your house, e.g. how
# room groups and individual fixtures are named. Keep it short: it is prefilled on every pass.
CTX = os.environ.get("HOUSE_NOTES",
                     "House notes: \"<Room> Lights\" is the group of every light in that room; a bare room name is "
                     "that room's main light; numbered names are individual fixtures.")

ENTITIES = []          # [{"names","domain","areas"}] — controllable, deduplicated by name
CAPTURED = None

def refresh_entities():
    global ENTITIES, CAPTURED
    try:
        d = json.load(urllib.request.urlopen(SURFACE_URL, timeout=20))
    except Exception as e:
        print(f"  ⚠️ surface refresh failed ({type(e).__name__}); keeping {len(ENTITIES)} entities", flush=True)
        return
    ents, cur = [], None
    for line in (d.get("system") or "").splitlines():
        m = re.match(r"^\s*- names: (.+)$", line)
        if m:
            cur = {"names": m.group(1).strip()}; ents.append(cur); continue
        m = re.match(r"^\s+(domain|areas): (.+)$", line)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2).strip()
    seen, out = set(), []
    for e in ents:
        if e.get("domain") in DOMAINS and e["names"] not in seen:
            seen.add(e["names"]); out.append(e)
    if out:
        ENTITIES, CAPTURED = out, d.get("captured_at")
        print(f"  surface: {len(out)} controllable entities (captured {CAPTURED})", flush=True)

def _refresher():
    while True:
        time.sleep(REFRESH_S); refresh_entities()

def desc(e):
    return e["names"] + (f" — a {e['domain']} in the {e['areas']}" if e.get("areas") else f" — a {e['domain']}")

def decide(state, question, options):
    row = {"id": "x", "state": state, "question": question, "options": [{"id": i, "description": d} for i, d in options]}
    r = B.score(MODEL, TOK, row, META, 4096)
    pr = sorted(zip(r["probabilities"], r["option_ids"]), reverse=True)
    return pr[0][1], pr[0][0], (pr[0][0] - pr[1][0]) if len(pr) > 1 else pr[0][0], r["forward_seconds"]

NUM = re.compile(r"(\d{1,3})")
def number_in(text):
    m = NUM.search(text.replace("percent", " ").replace("%", " "))
    return int(m.group(1)) if m else None

def run(text, thresholds=None):
    """thresholds: optional per-request overrides {action_p, margin, entity_p}. Absent keys keep the
    env defaults, so a bare {"text"} call uses those."""
    th = thresholds or {}
    ACTION_P = float(th.get("action_p", _ACTION_P)); MARGIN = float(th.get("margin", _MARGIN))
    ENTITY_P = float(th.get("entity_p", _ENTITY_P))
    t0 = time.time()
    st = f"{CTX}\nVoice request to the house assistant: \"{text}\""
    out = {"handled": False, "text": text}
    with LOCK:
        a, ap, _, _ = decide(st, "What does the request ask the house to do?", ACTIONS)
        out.update(action=a, action_p=round(ap, 3))
        if a == "not_a_device_command" or ap < ACTION_P:
            out["why"] = "not a device command" if a == "not_a_device_command" else f"action p {ap:.2f} < {ACTION_P}"
            out["latency_ms"] = int((time.time() - t0) * 1000); return out
        cands = [e for e in ENTITIES if e["domain"] in DOM[a]]
        if not cands:
            out["why"] = "no candidates"; out["latency_ms"] = int((time.time() - t0) * 1000); return out
        chunks = [cands[i:i + 14] for i in range(0, len(cands), 14)]; winners = []
        for ch in chunks:
            pick, p, m, _ = decide(st, "Which device does the request most likely mean?", [(f"e{j}", desc(e)) for j, e in enumerate(ch)])
            winners.append((ch[int(pick[1:])], p, m))
        if len(winners) > 1:
            pick, p, m, _ = decide(st, "Which device does the request most likely mean?", [(f"w{j}", desc(e)) for j, (e, _, _) in enumerate(winners)])
            ent = winners[int(pick[1:])][0]
        else:
            ent, p, m = winners[0]
    out.update(entity=ent["names"], entity_p=round(p, 3), margin=round(m, 3), candidates=len(cands))
    args = {"name": ent["names"]}
    if a == "set_brightness":
        n = number_in(text)
        if n is None:
            out["why"] = "no brightness number"; out["latency_ms"] = int((time.time() - t0) * 1000); return out
        args["brightness"] = max(0, min(100, n))
    if a == "set_temperature":
        n = number_in(text)
        if n is None:
            out["why"] = "no temperature number"; out["latency_ms"] = int((time.time() - t0) * 1000); return out
        args["temperature"] = n
    out["tool"] = {"name": TOOL[a], "arguments": args}
    if p < ENTITY_P or m < MARGIN:
        out["why"] = f"entity p {p:.2f} / margin {m:.2f} below {ENTITY_P}/{MARGIN}"
    else:
        out["handled"] = True
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return out

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": bool(ENTITIES), "entities": len(ENTITIES), "captured_at": CAPTURED,
                                    "thresholds": {"action_p": _ACTION_P, "margin": _MARGIN, "entity_p": _ENTITY_P}})
        if self.path == "/refresh":
            refresh_entities(); return self._send(200, {"entities": len(ENTITIES)})
        self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.path != "/decide":
            return self._send(404, {"error": "not found"})
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            text = str(d.get("text") or "").strip()
            if not text:
                return self._send(400, {"error": "text required"})
            th = d.get("thresholds") if isinstance(d.get("thresholds"), dict) else None
            r = run(text, th)
            if th:
                r["thresholds"] = th
            print(f"  {'ACT ' if r['handled'] else 'defer'} {text!r} -> {r.get('tool', {}).get('name', '-')} "
                  f"{r.get('entity', '-')!s} a={r.get('action_p')} p={r.get('entity_p')} m={r.get('margin')} "
                  f"{r['latency_ms']}ms {r.get('why', '')}", flush=True)
            self._send(200, r)
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

if __name__ == "__main__":
    refresh_entities()
    threading.Thread(target=_refresher, daemon=True).start()
    print("  warm:", run(os.environ.get("WARM_TEXT", "turn on the light"))["latency_ms"], "ms", flush=True)
    print(f"semif listening on :{PORT} — {len(ENTITIES)} entities, gguf {os.path.basename(GGUF)}", flush=True)
    ThreadingHTTPServer(("", PORT), H).serve_forever()
