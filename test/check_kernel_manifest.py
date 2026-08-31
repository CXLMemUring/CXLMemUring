#!/usr/bin/env python3
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile


def run(cira: str, source: str, manifest: pathlib.Path, output: pathlib.Path) -> None:
    with output.open("wb") as stream:
        subprocess.run(
            [
                cira,
                source,
                f"--cira-outline-rv64-kernels=manifest={manifest} module-id=monetdb",
            ],
            check=True,
            stdout=stream,
        )


def main() -> int:
    cira, source = sys.argv[1:3]
    with tempfile.TemporaryDirectory(prefix="cira-manifest-") as temp:
        root = pathlib.Path(temp)
        first = root / "first.json"
        second = root / "second.json"
        run(cira, source, first, root / "first.mlir")
        run(cira, source, second, root / "second.mlir")
        first_bytes = first.read_bytes()
        second_bytes = second.read_bytes()
        assert first_bytes == second_bytes, "kernel manifests are not byte-identical"

        manifest = json.loads(first_bytes)
        assert manifest["abi_version"] == 2
        assert len(manifest["kernels"]) == 1
        kernel = manifest["kernels"][0]
        identity = "monetdb | gdk/gdk_join.c | join_probe | 4451:1 | 7"
        expected_id = "0x" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        assert kernel["kernel_id"] == expected_id
        assert kernel["symbol"] == "__cira_rv64_" + expected_id[2:]
        assert kernel["source"] == "gdk/gdk_join.c:4451:1"
        assert kernel["query_tags"] == [9, 11]
        assert kernel["args_size"] == 16
        assert kernel["args"] == [
            {"name": "arg0", "offset": 0, "type": "arena_offset"},
            {"name": "arg1", "offset": 8, "type": "i32"},
        ]
        assert kernel["read_only"] is True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
