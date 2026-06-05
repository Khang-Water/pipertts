#!/usr/bin/env python3
"""Execute a short command on a Kaggle Jupyter proxy from KAGGLE_URL.

This uses the Jupyter kernel REST/WebSocket API and never prints the proxy URL.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning


requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", nargs="?", help="Python code to execute remotely")
    parser.add_argument("--code-file", type=Path, help="Read Python code from this local file")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--timeout-s", type=float, default=60.0)
    return parser.parse_args()


def load_kaggle_url(env_path: Path) -> str:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key.strip() == "KAGGLE_URL":
            return value.strip().strip("\"'")
    raise RuntimeError("KAGGLE_URL missing from .env")


def normalize_base(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    path = parsed.path.rstrip("/") + "/"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def ws_frame(payload: bytes) -> bytes:
    header = bytearray([0x81])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < (1 << 16):
        header.extend([0x80 | 126])
        header.extend(struct.pack("!H", length))
    else:
        header.extend([0x80 | 127])
        header.extend(struct.pack("!Q", length))
    mask = os.urandom(4)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return bytes(header) + mask + masked


def read_exact(sock: ssl.SSLSocket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = sock.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("websocket closed")
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(sock: ssl.SSLSocket) -> bytes:
    first = read_exact(sock, 2)
    opcode = first[0] & 0x0F
    length = first[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", read_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", read_exact(sock, 8))[0]
    if first[1] & 0x80:
        mask = read_exact(sock, 4)
        data = read_exact(sock, length)
        data = bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))
    else:
        data = read_exact(sock, length)
    if opcode == 8:
        raise ConnectionError("websocket close frame")
    if opcode == 9:
        sock.sendall(bytes([0x8A, len(data)]) + data)
        return read_frame(sock)
    return data


def open_ws(
    base_url: str,
    kernel_id: str,
    cookies: requests.cookies.RequestsCookieJar,
    timeout_s: float,
) -> ssl.SSLSocket:
    parsed = urlparse(base_url)
    ws_path = f"{parsed.path.rstrip('/')}/api/kernels/{kernel_id}/channels?session_id={uuid.uuid4()}"
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    cookie_header = "; ".join(f"{cookie.name}={cookie.value}" for cookie in cookies)
    raw_sock = socket.create_connection((parsed.hostname, parsed.port or 443), timeout=15)
    sock = ssl._create_unverified_context().wrap_socket(raw_sock, server_hostname=parsed.hostname)
    sock.settimeout(min(15.0, max(1.0, timeout_s)))
    request = (
        f"GET {ws_path} HTTP/1.1\r\n"
        f"Host: {parsed.netloc}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Origin: {parsed.scheme}://{parsed.netloc}\r\n"
        f"Cookie: {cookie_header}\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))
    response = b""
    while b"\r\n\r\n" not in response:
        response += sock.recv(4096)
    if b" 101 " not in response.split(b"\r\n", 1)[0]:
        raise ConnectionError("websocket upgrade failed")
    return sock


def execute(base_url: str, code: str, timeout_s: float) -> tuple[int, str]:
    session = requests.Session()
    session.verify = False
    session.get(base_url, timeout=15)
    kernel = session.post(base_url + "api/kernels", json={"name": "python3"}, timeout=15).json()
    kernel_id = kernel["id"]
    output: list[str] = []
    exit_code = 0
    try:
        sock = open_ws(base_url, kernel_id, session.cookies, timeout_s)
        msg_id = uuid.uuid4().hex
        message = {
            "header": {
                "msg_id": msg_id,
                "username": "pipertts",
                "session": uuid.uuid4().hex,
                "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
        }
        sock.sendall(ws_frame(json.dumps(message).encode("utf-8")))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                payload = read_frame(sock)
            except socket.timeout:
                continue
            event = json.loads(payload.decode("utf-8"))
            parent = event.get("parent_header", {})
            if parent.get("msg_id") != msg_id:
                continue
            msg_type = event.get("msg_type") or event.get("header", {}).get("msg_type")
            content = event.get("content", {})
            if msg_type == "stream":
                output.append(content.get("text", ""))
            elif msg_type in {"display_data", "execute_result"}:
                text = content.get("data", {}).get("text/plain")
                if text:
                    output.append(text + "\n")
            elif msg_type == "error":
                exit_code = 1
                output.append("\n".join(content.get("traceback", [])) + "\n")
            elif msg_type == "execute_reply":
                if content.get("status") != "ok":
                    exit_code = 1
                break
        else:
            exit_code = 124
            output.append("remote execution timed out\n")
        sock.close()
    finally:
        session.delete(base_url + f"api/kernels/{kernel_id}", timeout=15)
    return exit_code, "".join(output)


def main() -> int:
    args = parse_args()
    if args.code_file is not None:
        code = args.code_file.read_text(encoding="utf-8")
    elif args.code is not None:
        code = args.code
    else:
        raise SystemExit("provide code argument or --code-file")
    base_url = normalize_base(load_kaggle_url(args.env))
    try:
        exit_code, output = execute(base_url, code, args.timeout_s)
    except requests.RequestException as exc:
        print(f"kaggle request failed: {exc.__class__.__name__}")
        return 2
    except (ConnectionError, OSError) as exc:
        print(f"kaggle websocket failed: {exc.__class__.__name__}")
        return 2
    print(output, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
