#!/usr/bin/env python3

import pathlib
import subprocess
import sys
import tempfile


def run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")


def rv64_kernel_bodies(module: str) -> list[str]:
    lines = module.splitlines()
    bodies: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if "func.func private @__cira_rv64_" not in line or "cira.rv64_kernel" not in line:
            index += 1
            continue
        body = [line]
        depth = line.count("{") - line.count("}")
        index += 1
        while depth > 0 and index < len(lines):
            body.append(lines[index])
            depth += lines[index].count("{") - lines[index].count("}")
            index += 1
        if depth != 0:
            raise AssertionError("unterminated RV64 kernel body")
        bodies.append("\n".join(body))
    return bodies


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} CIRA REGION_SOURCE", file=sys.stderr)
        return 2
    cira, source = sys.argv[1:]

    with tempfile.TemporaryDirectory(prefix="cira-rv64-loads-") as temporary:
        root = pathlib.Path(temporary)
        formed = root / "formed.mlir"
        outlined = root / "outlined.mlir"
        manifest = root / "manifest.json"
        run([cira, source, "--cira-region-formation=require-profitable=false", "-o", str(formed)])
        run([
            cira,
            str(formed),
            f"--cira-outline-rv64-kernels=manifest={manifest} module-id=checked-load-test",
            "-o",
            str(outlined),
        ])

        formed_text = formed.read_text(encoding="utf-8")
        outlined_text = outlined.read_text(encoding="utf-8")
        bodies = rv64_kernel_bodies(outlined_text)
        prepared = formed_text.count("cira.prepared_root")
        installs = outlined_text.count("call @cira_device_install_cacheline")
        line_masks = outlined_text.count("arith.andi")
        checked = outlined_text.count("call @cira_device_checked_load")

        require(bodies, "no RV64 kernels were generated")
        require(prepared == 7, f"expected 7 prepared roots, found {prepared}")
        require(installs == prepared, "prepared roots and cache-line installs differ")
        require(line_masks == installs, "each cache-line install must have one 64-byte alignment mask")
        require(checked > installs, "ordinary and prepared loads must both use the checked-load helper")
        for body in bodies:
            require("call @cira_device_checked_load" in body, "RV64 kernel has no checked load")
            require("llvm.load" not in body, "RV64 kernel retains an unchecked LLVM load")
            require("memref.load" not in body, "RV64 kernel retains an unchecked memref load")

    print("RV64 checked-load and cache-line-install checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
