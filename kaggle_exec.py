#!/usr/bin/env python3
"""Execute code on the remote Kaggle kernel via Jupyter websocket API.

Usage:
    .venv/bin/python kaggle_exec.py "print('hello')"
    .venv/bin/python kaggle_exec.py --file script.py
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

import websocket

ENV_FILE = Path(__file__).parent / ".env"


def get_base_url() -> str:
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("KAGGLE_URL:"):
            return line.split(":", 1)[1].strip().rstrip("/")
    sys.exit("KAGGLE_URL not found in .env")


def get_kernel_id(base_url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(f"{base_url}/api/kernels", timeout=20) as r:
        kernels = json.load(r)
    if not kernels:
        sys.exit("No running kernels on remote server")
    return kernels[0]["id"]


def execute(code: str, timeout: float = 120.0) -> None:
    base_url = get_base_url()
    kernel_id = get_kernel_id(base_url)
    ws_url = base_url.replace("https://", "wss://") + f"/api/kernels/{kernel_id}/channels"

    ws = websocket.create_connection(ws_url, timeout=timeout)
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {
            "msg_id": msg_id,
            "username": "client",
            "session": uuid.uuid4().hex,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
    }))

    while True:
        msg = json.loads(ws.recv())
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        mtype = msg["msg_type"]
        content = msg["content"]
        if mtype == "stream":
            sys.stdout.write(content["text"])
            sys.stdout.flush()
        elif mtype in ("execute_result", "display_data"):
            print(content["data"].get("text/plain", ""))
        elif mtype == "error":
            print("\n".join(content["traceback"]), file=sys.stderr)
            ws.close()
            sys.exit(1)
        elif mtype == "status" and content["execution_state"] == "idle":
            break
    ws.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("code", nargs="?", help="Python code to run remotely")
    parser.add_argument("--file", help="Run a local .py file remotely")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    src = Path(args.file).read_text() if args.file else args.code
    if not src:
        parser.error("provide code or --file")
    execute(src, timeout=args.timeout)
