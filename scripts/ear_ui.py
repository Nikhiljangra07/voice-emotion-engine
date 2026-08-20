"""EAR UI — a one-click control panel for the live ear.

Local web page (127.0.0.1 only): route system audio, start/stop
listening, watch the trajectory live, and see the Affectogram the
moment a session ends. Past sessions browsable. Python stdlib only —
no new dependencies; the ear itself runs as a subprocess of the
existing scripts/live_ear.py with --headless --emit-jsonl.

Run:  venv/bin/python scripts/ear_ui.py          (any python works)
      -> opens http://127.0.0.1:8377
"""

import json
import queue
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EAR_PY = ROOT / ".venv_diar/bin/python"
OUT = ROOT / "out/live_ear"
PAGE = Path(__file__).with_name("ear_ui.html")
PORT = 8377

state = {"proc": None, "status": "idle", "source": "", "stem": None}
clients: list[queue.Queue] = []
lock = threading.Lock()


def broadcast(obj):
    msg = f"data: {json.dumps(obj)}\n\n".encode()
    with lock:
        for q in list(clients):
            q.put(msg)


def reader(proc):
    for line in proc.stdout:
        line = line.decode(errors="replace").rstrip()
        if line.startswith("EAR "):
            try:
                obj = json.loads(line[4:])
            except json.JSONDecodeError:
                continue
            if obj.get("event") == "done":
                state["stem"] = obj.get("stem")
            broadcast(obj)
        elif line:
            broadcast({"event": "log", "line": line[:300]})
    proc.wait()
    state.update(status="idle", proc=None)
    broadcast({"event": "stopped", "stem": state.get("stem")})


def start_ear(body):
    if state["proc"] is not None:
        return {"error": "already listening"}
    mode = body.get("mode", "device")
    cmd = [str(EAR_PY), "scripts/live_ear.py", "--headless", "--emit-jsonl"]
    if mode == "device":
        cmd += ["--device", str(int(body.get("device", 0)))]
        state["source"] = f"system audio (device {body.get('device', 0)})"
    elif mode == "simulate":
        path = body.get("path", "")
        if not (ROOT / path).exists() and not Path(path).exists():
            return {"error": f"file not found: {path}"}
        cmd += ["--simulate", path]
        state["source"] = f"simulate: {Path(path).name}"
    else:
        return {"error": f"unknown mode {mode}"}
    if body.get("duration"):
        cmd += ["--duration", str(float(body["duration"]))]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    state.update(proc=proc, status="listening", stem=None)
    threading.Thread(target=reader, args=(proc,), daemon=True).start()
    broadcast({"event": "started", "source": state["source"]})
    return {"ok": True}


def stop_ear():
    proc = state.get("proc")
    if proc is None:
        return {"error": "not listening"}
    proc.send_signal(signal.SIGINT)  # graceful: finish() -> Affectogram
    return {"ok": True}


def route_audio(on):
    exe = ROOT / "out/ear_multiout"
    if not exe.exists():
        return {"error": "out/ear_multiout not built — see DEMO.md"}
    r = subprocess.run([str(exe)] + ([] if on else ["revert"]),
                       capture_output=True, text=True, timeout=15)
    return {"ok": r.returncode == 0, "output": (r.stdout + r.stderr)[-400:]}


def sessions():
    rows = []
    for f in sorted(OUT.glob("*_traj.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)[:25]:
        stem = f.stem.replace("_traj", "")
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        speech = [r for r in data if r["emotion"] != "no-speech"]
        fams = [r["emotion"] for r in speech]
        rows.append({
            "stem": stem, "mtime": int(f.stat().st_mtime),
            "minutes": round(len(data) * 1.5 / 60, 1),
            "windows": len(data), "speech": len(speech),
            "dominant": max(set(fams), key=fams.count) if fams else "-",
            "affectogram": (OUT / f"{stem}_affectogram.png").exists(),
        })
    return rows


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            self._json({"status": state["status"],
                        "source": state["source"], "stem": state["stem"]})
        elif self.path == "/api/sessions":
            self._json(sessions())
        elif self.path.startswith("/api/affectogram/"):
            stem = Path(self.path.split("/api/affectogram/", 1)[1]).name
            f = OUT / f"{stem}_affectogram.png"
            if not f.exists():
                self._json({"error": "not rendered"}, 404)
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q: queue.Queue = queue.Queue()
            with lock:
                clients.append(q)
            try:
                self.wfile.write(b"data: {\"event\":\"hello\"}\n\n")
                self.wfile.flush()
                while True:
                    msg = q.get()
                    self.wfile.write(msg)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    if q in clients:
                        clients.remove(q)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if self.path == "/api/start":
            self._json(start_ear(body))
        elif self.path == "/api/stop":
            self._json(stop_ear())
        elif self.path == "/api/route":
            self._json(route_audio(bool(body.get("on", True))))
        else:
            self._json({"error": "not found"}, 404)


def main():
    if not EAR_PY.exists():
        sys.exit(f"missing {EAR_PY} — the ear venv is required")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"EAR UI at {url}  (Ctrl-C to quit)", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        if state.get("proc"):
            state["proc"].send_signal(signal.SIGINT)
        print("\nbye")


if __name__ == "__main__":
    main()
