"""HTTP server of a RunPod pod (started by hybrid/gpu/pod.sh).

GET  /<file>          a file of $OUT_DIR (log, status, results, checkpoints)
PUT  /in/<name>       stores the request body as $IN_DIR/<name> (or one chunk <name>.cKKKofNNN,
                      joined into <name> when all are in) when the X-Upload-Token header
                      equals $UPLOAD_TOKEN (a one-time value made by the workflow for this pod,
                      not a secret of the account): the workflow sends the models this way, since
                      the pod gets no GitHub token.
"""
from __future__ import annotations

import http.server
import os
import re
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "/workspace/out"))
IN = Path(os.environ.get("IN_DIR", "/workspace/in"))
TOKEN = os.environ.get("UPLOAD_TOKEN", "")


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(OUT), **k)

    def log_message(self, *a):   # quiet
        pass

    def do_PUT(self):
        m = re.fullmatch(r"/in/([A-Za-z0-9_.-]{1,80})", self.path)
        if not TOKEN or self.headers.get("X-Upload-Token") != TOKEN or not m:
            self.send_error(403)
            return
        n = int(self.headers.get("Content-Length", "0"))
        IN.mkdir(parents=True, exist_ok=True)
        tmp = IN / (m.group(1) + ".part")
        with open(tmp, "wb") as f:
            left = n
            while left > 0:
                b = self.rfile.read(min(left, 1 << 20))
                if not b:
                    break
                f.write(b)
                left -= len(b)
        if left:
            tmp.unlink(missing_ok=True)
            self.send_error(400, "short body")
            return
        name = m.group(1)
        c = re.fullmatch(r"(.+)\.c(\d{3})of(\d{3})", name)
        if not c:
            tmp.replace(IN / name)
        else:
            # one chunk of a large file (the proxy fails on bodies near 100 MB): kept until all
            # chunks are here, then joined atomically; a chunk sent twice just replaces itself
            tmp.replace(IN / name)
            base, n = c.group(1), int(c.group(3))
            parts = [IN / f"{base}.c{k:03d}of{n:03d}" for k in range(n)]
            if all(q.exists() for q in parts) and not (IN / base).exists():
                out = IN / (base + ".joining")
                with open(out, "wb") as f:
                    for q in parts:
                        f.write(q.read_bytes())
                out.replace(IN / base)
                for q in parts:
                    q.unlink(missing_ok=True)
        self.send_response(201)
        self.end_headers()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    http.server.ThreadingHTTPServer(("", 8000), H).serve_forever()
