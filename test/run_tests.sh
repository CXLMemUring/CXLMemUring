#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CIRA_BIN="$ROOT_DIR/build/bin/cira"
OUT_DIR="$ROOT_DIR/build/test_out"
mkdir -p "$OUT_DIR"

CLANGIR=""
if command -v clang >/dev/null 2>&1; then
  CLANGIR=`which clang`
fi

have_filecheck=0
if command -v FileCheck >/dev/null 2>&1; then
  have_filecheck=1
fi

ok=0; fail=0; xfail=0

run_chain() {
  local in_mlir="$1"; local base="$2"
  local remote="$OUT_DIR/${base}_remote.mlir"
  local hetero="$OUT_DIR/${base}_hetero.ll"
  "$CIRA_BIN" "$in_mlir" --rmem-search-remote -o "$remote"
  "$CIRA_BIN" "$remote" --convert-cira-to-llvm-hetero --convert-func-to-llvm --reconcile-unrealized-casts -o "$hetero"
}

try_run() {
  local name="$1"; shift
  if "$@"; then
    echo "[ OK ] $name"
    ok=$((ok+1))
  else
    echo "[FAIL] $name" >&2
    fail=$((fail+1))
  fi
}

echo "== Running MLIR pipeline tests =="
for mlir in \
  "$ROOT_DIR/test/mlir/types_i32.mlir" \
  "$ROOT_DIR/test/mlir/types_f64.mlir" \
  "$ROOT_DIR/test/mlir/indirect_store.mlir" \
  "$ROOT_DIR/test/mlir/mixed.mlir" \
  "$ROOT_DIR/test/mlir/bounds_var.mlir" \
  "$ROOT_DIR/test/mlir/nested_shapes.mlir" \
  "$ROOT_DIR/test/mlir/aliasing.mlir" \
  "$ROOT_DIR/test/mlir/aliasing_two.mlir" \
  "$ROOT_DIR/test/mlir/mixed_sizes.mlir" \
  "$ROOT_DIR/test/mlir/mixed_precision.mlir" \
  "$ROOT_DIR/test/mlir/deep_nest.mlir" \
  "$ROOT_DIR/test/mlir/deep_nest_branch.mlir" \
  "$ROOT_DIR/test/mlir/reduction.mlir" \
; do
  bname="$(basename "$mlir" .mlir)"
  try_run "${bname}" run_chain "$mlir" "$bname"
done

echo "== Running negative test (expected fail) =="
if "$CIRA_BIN" "$ROOT_DIR/test/mlir/negative_cast.mlir" --rmem-search-remote -o "$OUT_DIR/negative_remote.mlir" 2>"$OUT_DIR/negative.err"; then
  echo "[XFAIL] negative_cast expected to fail but passed" >&2
  xfail=$((xfail+1))
else
  if grep -q "arith.fptosi" "$OUT_DIR/negative.err"; then
    echo "[ OK ] negative_cast failed as expected"
    ok=$((ok+1))
  else
    echo "[FAIL] negative_cast failed but message mismatch" >&2
    fail=$((fail+1))
  fi
fi

echo "== FileCheck harness =="
if [[ $have_filecheck -eq 1 ]]; then
  # Remote extraction check
  "$CIRA_BIN" "$ROOT_DIR/test/mlir/check_remote_extract.mlir" --rmem-search-remote | FileCheck "$ROOT_DIR/test/mlir/check_remote_extract.mlir"
  echo "[ OK ] check_remote_extract"
  ok=$((ok+1))
  # Attribute tagging check
  "$CIRA_BIN" "$ROOT_DIR/test/mlir/check_cira_attrs.mlir" --convert-target-to-remote | FileCheck "$ROOT_DIR/test/mlir/check_cira_attrs.mlir"
  echo "[ OK ] check_cira_attrs"
  ok=$((ok+1))
else
  echo "[SKIP] FileCheck not found; skipping check_* tests"
fi

echo "== CIR frontend test =="
if [[ -n "$CLANGIR" ]]; then
    "$CLANGIR" -fclangir -emit-cir "$ROOT_DIR/test/cir/cir_frontend.c" -o "$OUT_DIR/cir_frontend.mlir"
    try_run "cir_frontend_search" "$CIRA_BIN" "$OUT_DIR/cir_frontend.mlir" --rmem-search-remote -o "$OUT_DIR/cir_frontend_remote.mlir"
    try_run "cir_frontend_lower" "$CIRA_BIN" "$OUT_DIR/cir_frontend_remote.mlir" --convert-cira-to-llvm-hetero --convert-func-to-llvm --reconcile-unrealized-casts -o "$OUT_DIR/cir_frontend_hetero.ll"
else
  echo "[SKIP] clangir not found; skipping CIR frontend test"
fi

echo "== Summary =="
echo "Passed: $ok  Failed: $fail  XFail: $xfail"
exit $(( fail > 0 ? 1 : 0 ))
