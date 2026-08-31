#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time


PASSES = [
    "cir-flatten-cfg",
    "cir-goto-solver",
    "cir-mlir-scf-prepare",
    "cir-lower-to-mlir",
    "cir-to-cira",
    "cira-region-formation",
    "cira-outline-rv64-kernels",
    "convert-cira-to-llvm-hetero",
    "convert-scf-to-cf",
    "cira-prune-unreachable-blocks",
    "convert-cf-to-llvm",
    "convert-to-llvm",
    "cse",
    "convert-ub-to-llvm",
    "reconcile-unrealized-casts",
    "cira-finalize-llvm-materializations",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_units(path: pathlib.Path) -> list[str]:
    units = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            units.append(line.split()[0])
    return units


def run(command: list[str], log: pathlib.Path, *, cwd: pathlib.Path, timeout: int = 900) -> float:
    start = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + shlex.join(command) + "\n")
        stream.flush()
        try:
            result = subprocess.run(command, cwd=cwd, text=True, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            fail(f"stage timed out after {timeout}s; see {log}")
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        fail(f"stage failed ({result.returncode}); see {log}\n{tail}")
    return elapsed


def configure(root: pathlib.Path, build: pathlib.Path) -> None:
    command = [
        "cmake", "-S", str(root / "bench/MonetDB"), "-B", str(build), "-G", "Ninja",
        "-DTESTING=OFF", "-DSTRICT=OFF", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=/usr/bin/clang", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    run(command, build / "cira-logs/configure.log", cwd=root)


def compile_entry(entry: dict, source_rel: str, root: pathlib.Path, work: pathlib.Path) -> dict:
    clang = pathlib.Path(os.environ.get("CLANGIR_BIN", root.parent / "clangir/build/bin/clang"))
    cira = pathlib.Path(os.environ.get("CIRA_BIN", root / "build/bin/cira"))
    mlir_translate = pathlib.Path(os.environ.get("MLIR_TRANSLATE_BIN", root.parent / "clangir/build/bin/mlir-translate"))
    llc = pathlib.Path(os.environ.get("LLC_BIN", root.parent / "clangir/build/bin/llc"))
    for tool in (clang, cira, mlir_translate, llc):
        if not tool.is_file() or not os.access(tool, os.X_OK):
            fail(f"required tool is not executable: {tool}")

    source = pathlib.Path(entry["file"]).resolve()
    command = entry.get("arguments") or shlex.split(entry["command"])
    arguments = list(command[1:])
    filtered: list[str] = []
    skip_next = False
    for argument in arguments:
        if skip_next:
            skip_next = False
            continue
        if argument in ("-o", "-MF", "-MT", "-MQ"):
            skip_next = True
            continue
        if argument in ("-c", str(source)):
            continue
        if argument.startswith("-W") or argument.startswith("-O") or argument.startswith("-fstack-protector"):
            continue
        filtered.append(argument)

    stem = pathlib.Path(source_rel).with_suffix("")
    unit_dir = work / stem
    unit_dir.mkdir(parents=True, exist_ok=True)
    cir = unit_dir / "input.cir"
    lowered = unit_dir / "host.llvm.mlir"
    llvm_ir = unit_dir / "host.ll"
    host_object = unit_dir / "host.o"
    manifest = unit_dir / "kernels.json"
    logs = unit_dir / "logs"

    timings = {}
    timings["frontend_seconds"] = run(
        [str(clang), "-fclangir", "-emit-cir", "-S", "-fno-strict-aliasing", "-O1",
         "-DCIRA_MONETDB_SHARED_ARENA=1", *filtered, str(source), "-o", str(cir)],
        logs / "01-frontend.log", cwd=root,
    )
    if not cir.is_file() or cir.stat().st_size == 0:
        fail(f"frontend produced no CIR for {source_rel}")
    pipeline = "builtin.module(" + ",".join([
        "cir-flatten-cfg",
        "cir-goto-solver",
        "cir-mlir-scf-prepare",
        "cir-lower-to-mlir",
        "cir-to-cira",
        "cira-region-formation{require-profitable=false}",
        f"cira-outline-rv64-kernels{{manifest={manifest} module-id=monetdb}}",
        "convert-cira-to-llvm-hetero",
        "convert-scf-to-cf",
        "cira-prune-unreachable-blocks",
        "convert-cf-to-llvm",
        "convert-to-llvm",
        "cse",
        "convert-ub-to-llvm",
        "reconcile-unrealized-casts",
        "cira-finalize-llvm-materializations",
    ]) + ")"
    timings["cira_pipeline_seconds"] = run(
        [str(cira), "-mlir-print-op-on-diagnostic=false", "-allow-unregistered-dialect",
         str(cir), f"--pass-pipeline={pipeline}", "-o", str(lowered)],
        logs / "02-cira-pipeline.log", cwd=root, timeout=1800,
    )
    timings["llvm_translation_seconds"] = run(
        [str(mlir_translate), "--allow-unregistered-dialect", "--mlir-to-llvmir", str(lowered),
         "-o", str(llvm_ir)],
        logs / "03-llvm-translation.log", cwd=root,
    )
    timings["object_codegen_seconds"] = run(
        [str(llc), "-O2", "-filetype=obj", "-relocation-model=pic", "-disable-cgp",
         "-mtriple=x86_64-unknown-linux-gnu", str(llvm_ir), "-o", str(host_object)],
        logs / "04-object-codegen.log", cwd=root,
    )

    kernel_ids: list[str] = []
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("abi_version") != 2:
            fail(f"{source_rel} produced a non-v2 kernel manifest")
        kernel_ids = [kernel["kernel_id"] for kernel in data.get("kernels", [])]
    return {
        "source": source_rel,
        "mode": "cira-hetero",
        "fallback_allowed": False,
        "compiler": str(clang),
        "passes": PASSES,
        "object": str(host_object),
        "kernel_ids": kernel_ids,
        "artifacts": {
            "cir": str(cir), "host_llvm_mlir": str(lowered), "llvm_ir": str(llvm_ir),
            "host_object": str(host_object), "manifest": str(manifest),
        },
        "timings": timings,
    }


def check_units(root: pathlib.Path, units_path: pathlib.Path, contract_path: pathlib.Path) -> None:
    native_build = root / "build/monetdb-cira-native"
    configure(root, native_build)
    compile_commands = json.loads((native_build / "compile_commands.json").read_text(encoding="utf-8"))
    by_source = {str(pathlib.Path(entry["file"]).resolve()): entry for entry in compile_commands}
    records = []
    work = root / "build/monetdb_pipeline/readiness"
    for source_rel in read_units(units_path):
        absolute = str((root / "bench/MonetDB" / source_rel).resolve())
        entry = by_source.get(absolute)
        if entry is None:
            fail(f"native MonetDB configuration omitted targeted unit {source_rel}")
        print(f"[monetdb-cira] checking {source_rel}", flush=True)
        records.append(compile_entry(entry, source_rel, root, work))
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps({"schema_version": 1, "units": records}, indent=2) + "\n",
                             encoding="utf-8")


def output_path(entry: dict) -> pathlib.Path:
    command = entry.get("arguments") or shlex.split(entry["command"])
    try:
        output = command[command.index("-o") + 1]
    except (ValueError, IndexError):
        fail(f"compile command has no object output: {entry['file']}")
    path = pathlib.Path(output)
    if not path.is_absolute():
        path = pathlib.Path(entry["directory"]) / path
    return path.resolve()


def compile_native_overlay(entry: dict, root: pathlib.Path, log: pathlib.Path) -> pathlib.Path:
    command = entry.get("arguments") or shlex.split(entry["command"])
    command = list(command)
    command[0] = "/usr/bin/clang"
    command.insert(1, "-DCIRA_MONETDB_SHARED_ARENA=1")
    run(command, log, cwd=pathlib.Path(entry["directory"]))
    return output_path(entry)


def build_server(root: pathlib.Path, units_path: pathlib.Path, contract_path: pathlib.Path) -> None:
    check_units(root, units_path, contract_path)
    native_build = root / "build/monetdb-cira-native"
    parallel = str(max(1, min(os.cpu_count() or 1, 16)))
    run(["cmake", "--build", str(native_build), "--target", "mserver5", "--parallel", parallel],
        native_build / "cira-logs/08-native-server.log", cwd=root, timeout=1800)

    compile_commands = json.loads((native_build / "compile_commands.json").read_text(encoding="utf-8"))
    by_source = {str(pathlib.Path(entry["file"]).resolve()): entry for entry in compile_commands}
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    records = {record["source"]: record for record in contract["units"]}
    for source_rel in read_units(units_path):
        entry = by_source[str((root / "bench/MonetDB" / source_rel).resolve())]
        destination = output_path(entry)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(records[source_rel]["artifacts"]["host_object"], destination)
        os.utime(destination, None)
        records[source_rel]["object"] = str(destination)

    for source_rel in ("gdk/gdk_heap.c", "gdk/gdk_utils.c"):
        entry = by_source[str((root / "bench/MonetDB" / source_rel).resolve())]
        compile_native_overlay(entry, root, native_build / "cira-logs" / (pathlib.Path(source_rel).stem + ".log"))

    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    run(["cmake", "--build", str(native_build), "--target", "mserver5", "--parallel", parallel],
        native_build / "cira-logs/09-relink-server.log", cwd=root, timeout=1800)

    candidates = [native_build / "tools/mserver/mserver5", native_build / "bin/mserver5"]
    server = next((candidate for candidate in candidates if candidate.is_file()), None)
    if server is None:
        fail("native MonetDB build did not produce mserver5")
    destination = root / "bin/monetdb/x86_64-unknown-linux-gnu/mserver5"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(server, destination)
    destination.chmod(destination.stat().st_mode | 0o111)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-units", action="store_true")
    parser.add_argument("--build-server", action="store_true")
    parser.add_argument("--units", required=True, type=pathlib.Path)
    parser.add_argument("--contract", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.check_units == args.build_server:
        parser.error("select exactly one of --check-units or --build-server")
    root = pathlib.Path(__file__).resolve().parents[1]
    if args.check_units:
        check_units(root, args.units.resolve(), args.contract.resolve())
    else:
        build_server(root, args.units.resolve(), args.contract.resolve())
    print(f"[monetdb-cira] contract: {args.contract.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"monetdb_cira_compile.py: {error}", file=sys.stderr)
        raise SystemExit(1)
