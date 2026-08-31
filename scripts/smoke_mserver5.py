#!/usr/bin/env python3

import argparse
import pathlib
import signal
import socket
import subprocess
import sys
import time


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_until_listening(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = process.poll()
        if status is not None:
            raise RuntimeError(f"mserver5 exited before becoming ready (status {status})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"mserver5 did not listen on port {port} within {timeout:g}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start, observe, and cleanly stop mserver5")
    parser.add_argument("--server", required=True, type=pathlib.Path)
    parser.add_argument("--dbpath", required=True, type=pathlib.Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    server = args.server.resolve()
    dbpath = args.dbpath.resolve()
    if not server.is_file() or not server.stat().st_mode & 0o111:
        raise RuntimeError(f"server is not executable: {server}")
    if args.timeout <= 0:
        raise RuntimeError("--timeout must be positive")

    dbpath.mkdir(parents=True, exist_ok=True)
    if (dbpath / ".started").exists():
        raise RuntimeError(f"database already appears live: {dbpath}")

    log_dir = dbpath.parent / f"{dbpath.name}-smoke-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    port = reserve_port()
    command = [
        str(server),
        f"--dbpath={dbpath}",
        "--set", "mapi_listenaddr=127.0.0.1",
        "--set", f"mapi_port={port}",
    ]

    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        try:
            wait_until_listening(process, port, args.timeout)
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

        process.send_signal(signal.SIGTERM)
        try:
            status = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("mserver5 did not stop after SIGTERM")
        if status != 0:
            raise RuntimeError(f"mserver5 shutdown returned status {status}")

    print(f"MSERVER5_SMOKE PASS port={port}")
    print(f"stdout={stdout_path}")
    print(f"stderr={stderr_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"smoke_mserver5.py: {error}", file=sys.stderr)
        raise SystemExit(1)
