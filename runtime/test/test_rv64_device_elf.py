#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def run(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None, check=False)
    if result.returncode != 0:
        output = result.stdout or ""
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{output}")
    return result.stdout or ""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    if len(sys.argv) != 7:
        print(f"usage: {sys.argv[0]} DEVICE_SOURCE BUILD_DIR GCC READELF NM OBJDUMP", file=sys.stderr)
        return 2
    source, build, compiler, readelf, nm, objdump = map(Path, sys.argv[1:])
    manifest = source / "test" / "two-kernel-manifest.json"
    kernels = source / "test" / "two-kernels.c"
    run([
        "cmake", "-S", str(source), "-B", str(build),
        f"-DCMAKE_C_COMPILER={compiler}", f"-DCMAKE_ASM_COMPILER={compiler}",
        f"-DCIRA_KERNEL_MANIFEST={manifest}", f"-DCIRA_KERNEL_SOURCES={kernels}",
    ])
    run(["cmake", "--build", str(build), "--parallel"])

    elf = build / "cira-rv64-device.elf"
    header = run([str(readelf), "-h", str(elf)], capture=True)
    symbols = run([str(readelf), "-Ws", str(elf)], capture=True)
    undefined = run([str(nm), "-u", str(elf)], capture=True)
    disassembly = run([str(objdump), "-d", str(elf)], capture=True)
    require("Class:                             ELF64" in header, "device image is not ELF64")
    require("Machine:                           RISC-V" in header, "device image is not RISC-V")
    require("Entry point address:               0x10000" in header, "device entry point is not _start")
    for symbol in (
        "_start", "cira_device_main", "cira_device_checked_load", "cira_device_install_cacheline",
        "__cira_rv64_0000000000000002", "__cira_rv64_0000000000000009",
    ):
        require(symbol in symbols, f"device ELF is missing {symbol}")
    require(not undefined.strip(), f"device ELF has undefined symbols:\n{undefined}")
    require("0005000b" in disassembly or ".insn" in disassembly,
            "device ELF does not contain the custom cache-install instruction")
    print("RV64 no-libc device ELF checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
