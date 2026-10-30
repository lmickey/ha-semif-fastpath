#!/usr/bin/env python3
"""semif_proxy — an Ollama-compatible proxy that settles simple Home Assistant voice commands
without running the LLM. Pure stdlib Python 3.10+, one file.

Point HA's Ollama integration at this proxy instead of at Ollama. Everything is relayed unchanged,
except /api/chat requests carrying HA's tool surface (tools named Hass*), where two rules apply.
Both FAIL OPEN: any doubt, timeout or error and the request goes to the LLM as if this were absent.

  1. FRESH COMMAND → semif first. If the last message is a fresh, single user command, ask
     semif_service's /decide. A confident answer comes back to HA as the LLM's tool call; HA runs it.
     Measured: "turn off <a light>" 0.58 s end to end, was 5-10 s through the LLM.
  2. AFTER ONE SUCCESSFUL DEVICE ACTION → reply with a single space. HA then makes a second LLM call
     only to say "Done"; answering it with whitespace skips that round trip, and HA skips TTS on a
     blank reply (assist_pipeline tests tts_input.strip()), so the satellite's done chime is the
     confirmation. Applies whether semif or the LLM chose the call.
     🔥 A SPACE, not "". HA drops an empty assistant message, re-asks up to its iteration cap and
        then says "Unable to get response".

TURNS semif MUST NOT TAKE (each one bit us; semif_eligible() encodes them, --selftest checks them)
  · The ANSWER to a clarifying question. "Turn off the lamp" → LLM: "which lamp?" → "the one by the
    desk". semif sees only that text and has no idea the verb was OFF; it once emitted HassTurnOn.
    If the nearest assistant message is a question with no tool call, the LLM takes it.
  · COMPOUND commands. "turn off the desk lamp and the fan" → semif picks ONE device, HA does it,
    the blank reply says nothing, and the other device stays on. "and" / "then" / "also" → LLM.
    (Not ",": Whisper writes "Hey Nabu, turn …".)
  · The WAKE WORD's tail. If your satellite keeps pre-roll audio, Whisper may transcribe the wake
    word; set WAKE_WORDS so it is stripped before semif sees the text.

Only models listed in INTERCEPT_MODELS are intercepted (empty = every model). Use that to keep a
second model name that always reaches the real LLM — for regression tests that must grade the LLM,
not semif. A request with header X-Semif: off is never intercepted.

GET /debug/ha-surface returns the last HA system prompt seen (it lists your exposed entities);
semif_service reads its device list from here. ⚠️ Do not expose this endpoint beyond your LAN.

Env: OLLAMA_URL (upstream, http://localhost:11435), SEMIF_URL (http://localhost:8090; empty =
     layer off, proxy only), PORT (11434), SEMIF_TIMEOUT (1.5 s), INTERCEPT_MODELS (comma list),
     WAKE_WORDS (comma list, e.g. "hey nabu,okay nabu"), BLANK_REPLY (1), ACTION_P / MARGIN /
     ENTITY_P (sent with each /decide call; omit to use semif's own defaults).

    python3 semif_proxy.py              # serve
    python3 semif_proxy.py --selftest   # turn rules + an end-to-end run against stub backends
"""
import json, os, re, sys, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11435").rstrip("/")
SEMIF_URL = os.environ.get("SEMIF_URL", "http://localhost:8090").rstrip("/")
PORT = int(os.environ.get("PORT", "11434"))
SEMIF_TIMEOUT = float(os.environ.get("SEMIF_TIMEOUT", "1.5"))
INTERCEPT = {m.strip().lower() for m in os.environ.get("INTERCEPT_MODELS", "").split(",") if m.strip()}
BLANK_REPLY = os.environ.get("BLANK_REPLY", "1") not in ("0", "false", "no")
THRESHOLDS = {k.lower(): float(os.environ[k]) for k in ("ACTION_P", "MARGIN", "ENTITY_P") if os.environ.get(k)}
TIMEOUT = 900       # a cold model load can take minutes; the relay must not give up first

DEVICE_TOOLS = {"HassTurnOn", "HassTurnOff", "HassLightSet", "HassClimateSetTemperature", "HassSetPosition", "HassStopMoving"}
COMPOUND_RE = re.compile(r"\b(and|then|also)\b", re.I)


def _wake_re(words):
    alts = [r"\s+".join(map(re.escape, w.split())) for w in words.split(",") if w.strip()]
    return re.compile(r"^\s*(?:" + "|".join(alts) + r")\b[\s,.!?]*", re.I) if alts else None

WAKE_RE = _wake_re(os.environ.get("WAKE_WORDS", ""))
SURFACE = {"system": None, "captured_at": None}


def strip_wake(text, wake_re=None):
    r = wake_re or WAKE_RE
    return r.sub("", text, count=1) if r else text


def is_ha_surface(d):
    names = [str((t.get("function") or {}).get("name")) for t in (d.get("tools") or []) if isinstance(t, dict)]
    return any(n.startswith("Hass") for n in names)


def semif_eligible(msgs, wake_re=None):
    """Is the last message a FRESH, SINGLE user command semif may settle? → (ok, why)."""
    if not msgs or msgs[-1].get("role") != "user":
        return False, "last message not from the user"
    text = msgs[-1].get("content")
    if not isinstance(text, str) or not text.strip():
        return False, "empty"
    for m in reversed(msgs[:-1]):
        r = m.get("role")
        if r == "assistant":
            if not m.get("tool_calls") and (m.get("content") or "").strip().endswith("?"):
                return False, "follow-up to a clarifying question"
            break
        if r == "user":
            break
    if COMPOUND_RE.search(strip_wake(text, wake_re)):
        return False, "compound command"
    return True, ""


def blank_after_device_action(msgs):
    """The turn right after ONE device tool ran and succeeded → its name (reply with nothing). Else None."""
    if not msgs or msgs[-1].get("role") != "tool":
        return None
    prev = None
    for m in reversed(msgs[:-1]):
        if m.get("role") == "assistant":
            prev = m; break
        if m.get("role") == "user":
            return None
    calls = (prev or {}).get("tool_calls") or []
    if len(calls) != 1:
        return None
    name = (calls[0].get("function") or {}).get("name")
    if name not in DEVICE_TOOLS:
        return None
    try:
        res = json.loads(msgs[-1].get("content") or "{}")
    except Exception:
        return None
    if not isinstance(res, dict) or res.get("response_type") != "action_done":
        return None
    if (res.get("data") or {}).get("failed"):
        return None
    return name


def semif_decide(text):
    """Ask semif_service. None on any failure — the turn then goes to the LLM."""
    if not SEMIF_URL:
        return None
    body = {"text": text, **({"thresholds": THRESHOLDS} if THRESHOLDS else {})}
    try:
        req = urllib.request.Request(SEMIF_URL + "/decide", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=SEMIF_TIMEOUT) as r:
            return json.load(r)
    except Exception as e:
        print(f"  semif unavailable ({type(e).__name__}); LLM handles it", flush=True)
        return None


def ollama_chat_reply(model, content="", tool_calls=None, stream=True):
    """An /api/chat reply as Ollama would send it: NDJSON when streaming, one object otherwise."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())
    done = {"model": model, "created_at": now, "message": {"role": "assistant", "content": ""}, "done_reason": "stop",
            "done": True, "total_duration": 1, "load_duration": 0, "prompt_eval_count": 0, "prompt_eval_duration": 0,
            "eval_count": 0, "eval_duration": 0}
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if stream:
        first = {"model": model, "created_at": now, "message": msg, "done": False}
        return (json.dumps(first) + "\n" + json.dumps(done) + "\n").encode()
    done["message"] = msg
    return json.dumps(done).encode()


def decision_layer(d, bypass=False):
    """Bytes of a synthesized reply for a turn we can settle without the LLM, else None."""
    msgs = d.get("messages") or []
    model = d.get("model") or ""
    stream = d.get("stream", True)
    if bypass or not msgs or (INTERCEPT and model.lower() not in INTERCEPT):
        return None
    name = blank_after_device_action(msgs)
    if name:
        if not BLANK_REPLY:
            return None
        print(f"  BLANK reply after {name} succeeded (no LLM round trip)", flush=True)
        return ollama_chat_reply(model, " ", None, stream)
    ok, why = semif_eligible(msgs)
    if not ok:
        if why not in ("last message not from the user", "empty"):
            print(f"  skip ({why}): {str(msgs[-1].get('content'))[:60]!r} -> LLM", flush=True)
        return None
    text = strip_wake(msgs[-1]["content"]).strip()
    if not text:
        return None
    r = semif_decide(text)
    if not r:
        return None
    if not r.get("handled"):
        print(f"  defer {text!r}: {r.get('why', '?')} ({r.get('latency_ms')} ms) -> LLM", flush=True)
        return None
    tool = r["tool"]
    print(f"  ACT {text!r} -> {tool['name']} {json.dumps(tool['arguments'])} "
          f"a={r.get('action_p')} p={r.get('entity_p')} m={r.get('margin')} ({r.get('latency_ms')} ms)", flush=True)
    return ollama_chat_reply(model, "", [{"function": {"name": tool["name"], "arguments": tool["arguments"]}}], stream)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _relay(self, body=None):
        """Forward to Ollama and stream the answer back chunk by chunk, so token streaming survives."""
        hdrs = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "accept", "authorization")}
        req = urllib.request.Request(OLLAMA_URL + self.path, data=body, headers=hdrs, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        except urllib.error.HTTPError as e:
            resp = e
        except Exception as e:
            return self._send(502, json.dumps({"error": f"upstream: {type(e).__name__}: {e}"}).encode())
        with resp:
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else None

    def do_GET(self):
        if self.path == "/debug/ha-surface":
            return self._send(200 if SURFACE["system"] else 404, json.dumps(SURFACE).encode())
        self._relay()

    def do_HEAD(self):
        self._relay()

    def do_DELETE(self):
        self._relay(self._body())

    def do_PUT(self):
        self._relay(self._body())

    def do_POST(self):
        body = self._body()
        if self.path == "/api/chat" and body:
            try:
                d = json.loads(body)
            except Exception:
                d = None
            if isinstance(d, dict) and is_ha_surface(d):
                sys_msg = next((m.get("content") for m in d.get("messages") or [] if m.get("role") == "system"), None)
                if sys_msg:
                    SURFACE.update(system=sys_msg, captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                try:
                    reply = decision_layer(d, bypass=self.headers.get("X-Semif", "").lower() == "off")
                except Exception as e:
                    print(f"  decision layer error ({type(e).__name__}: {e}); LLM handles it", flush=True)
                    reply = None
                if reply is not None:
                    return self._send(200, reply, "application/x-ndjson")
        self._relay(body)


# ── selftest ─────────────────────────────────────────────────────────────────────────────────
def _selftest():
    U = lambda t: {"role": "user", "content": t}
    A = lambda t, calls=None: {"role": "assistant", "content": t, **({"tool_calls": calls} if calls else {})}
    S = {"role": "system", "content": "x"}
    call = [{"function": {"name": "HassTurnOff", "arguments": {"name": "Desk Lamp"}}}]
    ok_res = {"role": "tool", "content": '{"response_type": "action_done", "data": {"success": [], "failed": []}}'}
    wr = _wake_re("hey nabu,okay nabu")
    cases = [
        ([S, U("Turn off the desk lamp.")], True, ""),
        ([S, U("Turn off the lamp."), A("", call), ok_res, A("Which lamp do you mean?"), U("The one by the desk.")], False, "follow-up to a clarifying question"),
        ([S, U("Turn on the fan."), A("", call), ok_res, A(" "), U("Turn off the desk lamp.")], True, ""),
        ([S, U("Turn off the desk lamp and the fan.")], False, "compound command"),
        ([S, U("Turn off the lamp, then the fan")], False, "compound command"),
        ([S, U("Turn on the bed stand.")], True, ""),                       # 'and' inside a word is not compound
        ([S, U(" Hey Nabu, turn off the desk lamp.")], True, ""),          # wake tail + Whisper's comma
        ([S, U("Turn off the desk lamp."), A("", call), ok_res], False, "last message not from the user"),
        ([S, U("   ")], False, "empty"),
    ]
    for msgs, ok, why in cases:
        got = semif_eligible(msgs, wr)
        assert got == (ok, why), (msgs[-1], got, (ok, why))
    assert strip_wake("Okay Nabu. What time is it?", wr) == "What time is it?"
    assert blank_after_device_action([S, U("x"), A("", call), ok_res]) == "HassTurnOff"
    failed = {"role": "tool", "content": '{"response_type": "action_done", "data": {"failed": [{"name": "x"}]}}'}
    assert blank_after_device_action([S, U("x"), A("", call), failed]) is None
    assert blank_after_device_action([S, U("x"), A("", call + call), ok_res]) is None
    print(f"  [ok] turn rules ({len(cases) + 4} cases)")

    # End to end: stub Ollama + stub semif on loopback, the proxy between them.
    global OLLAMA_URL, SEMIF_URL
    seen = []

    class StubOllama(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0)); seen.append(self.path)
            b = ollama_chat_reply("llm", "from the LLM")
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    class StubSemif(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def do_POST(self):
            t = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))["text"]
            r = ({"handled": True, "tool": {"name": "HassTurnOff", "arguments": {"name": "Desk Lamp"}}}
                 if "desk lamp" in t else {"handled": False, "why": "margin 0.1 < 0.6"})
            b = json.dumps(r).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), c) for c in (StubOllama, StubSemif, H)]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    OLLAMA_URL, SEMIF_URL = (f"http://127.0.0.1:{s.server_address[1]}" for s in servers[:2])
    proxy = f"http://127.0.0.1:{servers[2].server_address[1]}"
    tools = [{"type": "function", "function": {"name": "HassTurnOff"}}]
    system = {"role": "system", "content": "- names: Desk Lamp\n  domain: light\n  areas: Office"}

    def chat(msgs, hdrs=None):
        req = urllib.request.Request(proxy + "/api/chat", data=json.dumps({"model": "m", "messages": msgs, "tools": tools}).encode(),
                                     headers={"Content-Type": "application/json", **(hdrs or {})})
        return [json.loads(l) for l in urllib.request.urlopen(req, timeout=5).read().splitlines() if l.strip()]

    r = chat([system, U("turn off the desk lamp")])
    assert r[0]["message"]["tool_calls"][0]["function"]["name"] == "HassTurnOff" and not seen, r
    r = chat([system, U("turn off the desk lamp"), A("", call), ok_res])
    assert r[0]["message"]["content"] == " " and not seen, r
    r = chat([system, U("dim something")])
    assert r[0]["message"]["content"] == "from the LLM" and seen == ["/api/chat"], r
    r = chat([system, U("turn off the desk lamp")], {"X-Semif": "off"})
    assert r[0]["message"]["content"] == "from the LLM", r
    surf = json.load(urllib.request.urlopen(proxy + "/debug/ha-surface", timeout=5))
    assert "Desk Lamp" in surf["system"], surf
    for s in servers:
        s.shutdown()
    print("  [ok] end to end: semif act, blank reply, defer to LLM, X-Semif bypass, surface capture")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest(); sys.exit(0)
    print(f"semif_proxy :{PORT} -> {OLLAMA_URL}  semif={SEMIF_URL or 'off'}  "
          f"intercept={','.join(sorted(INTERCEPT)) or 'all models'}", flush=True)
    ThreadingHTTPServer(("", PORT), H).serve_forever()
