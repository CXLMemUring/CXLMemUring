#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def kernel(kernel_id: int, args_size: int = 32, read_only: bool = True) -> dict:
    return {
        "kernel_id": f"0x{kernel_id:016x}",
        "symbol": f"__cira_rv64_{kernel_id:016x}",
        "source": "dispatch-test.c:1:1",
        "query_tags": [1],
        "args_size": args_size,
        "args": [
            {"name": "arg0", "offset": 0, "type": "arena_offset"},
            {"name": "arg1", "offset": 8, "type": "i64"},
            {"name": "arg2", "offset": 16, "type": "arena_offset"},
            {"name": "arg3", "offset": 24, "type": "i32"},
        ],
        "read_only": read_only,
    }


def invoke(generator: Path, manifest: dict, directory: Path) -> subprocess.CompletedProcess[str]:
    manifest_path = directory / "kernels.json"
    output_path = directory / "dispatch.h"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(generator), str(manifest_path), str(output_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} GEN_DISPATCH_TABLE", file=sys.stderr)
        return 2
    generator = Path(sys.argv[1])

    with tempfile.TemporaryDirectory(prefix="cira-rv64-dispatch-") as temporary:
        directory = Path(temporary)
        manifest = {"abi_version": 2, "kernels": [kernel(9, 32), kernel(2, 32)]}
        result = invoke(generator, manifest, directory)
        require(result.returncode == 0, f"valid manifest failed: {result.stderr}")
        generated = (directory / "dispatch.h").read_text(encoding="utf-8")
        first = "{ UINT64_C(0x0000000000000002), __cira_rv64_0000000000000002, 32 },"
        second = "{ UINT64_C(0x0000000000000009), __cira_rv64_0000000000000009, 32 },"
        require(first in generated and second in generated, "generated table is missing typed entries")
        require(generated.index(first) < generated.index(second), "dispatch entries are not sorted by kernel ID")

        duplicate = {"abi_version": 2, "kernels": [kernel(2), kernel(2)]}
        result = invoke(generator, duplicate, directory)
        require(result.returncode != 0 and "duplicate kernel ID" in result.stderr,
                "duplicate kernel IDs must fail")

        mismatched = kernel(3, 40)
        result = invoke(generator, {"abi_version": 2, "kernels": [mismatched]}, directory)
        require(result.returncode != 0 and "args_size" in result.stderr,
                "argument-layout size mismatch must fail")

        writable = kernel(4, read_only=False)
        result = invoke(generator, {"abi_version": 2, "kernels": [writable]}, directory)
        require(result.returncode != 0 and "read-only" in result.stderr,
                "non-read-only kernels must fail")

    print("RV64 dispatch-table generator checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
