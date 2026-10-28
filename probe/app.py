import http.server
import os
import subprocess

BW = "bwrap --die-with-parent --ro-bind /usr /usr --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/bin /bin --proc /proc --dev /dev --tmpfs /tmp"
CMDS = [
    "uname -r",
    "id",
    "cat /proc/sys/user/max_user_namespaces",
    "unshare -Ur true && echo USERNS_OK",
    "bwrap --version",
    BW + " --unshare-all echo BWRAP_FULL_OK",
    BW + " --unshare-net echo BWRAP_NET_ONLY_OK",
    BW + " echo BWRAP_BARE_OK",
    BW + " --unshare-all /usr/local/bin/python3 -c 'print(1+1)'",
    "free -m",
]


def run(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        return "$ " + cmd + "\nexit=" + str(r.returncode) + "\n" + r.stdout + r.stderr + "\n"
    except Exception as exc:
        return "$ " + cmd + "\nERROR " + repr(exc) + "\n"


RESULT = "".join(run(c) for c in CMDS)
print(RESULT, flush=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = RESULT.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


http.server.HTTPServer(("0.0.0.0", int(os.environ.get("PORT", "10000"))), Handler).serve_forever()