#!/bin/bash
# Iterative PGO compilation for CXL Type2 GPU offloading
# Usage: ./scripts/pgo_iterate.sh <input.mlir> [iterations]
set -e

CIRA="./build/bin/cira"
PIPELINE="builtin.module(gpu-offload-decision,gpu-kernel-gen,gpu-memory-opt,gpu-runtime-lowering"
TYPE2_SRC="/root/ia780i_type2_delay_buffer/Type2GpuDevice.cpp"
TYPE2_INC="/root/ia780i_type2_delay_buffer"
CXLMEM_INC="./include"
PROFILE="pgo_profile.json"
INPUT="${1:?Usage: $0 <input.mlir> [iterations]}"
ITERS="${2:-3}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CXL Type2 GPU — Iterative PGO + Prefetch Optimization     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Input:      $INPUT"
echo "Iterations: $ITERS"
echo ""

rm -f "$PROFILE"

for iter in $(seq 1 "$ITERS"); do
    echo "━━━ Iteration $iter/$ITERS ━━━"

    if [ "$iter" -eq 1 ]; then
        # First iteration: instrument
        echo "  [1/4] Compiling (PGO=instrument)..."
        $CIRA -pass-pipeline="${PIPELINE},type2-gpu-codegen{pgo=instrument})" \
            "$INPUT" -o /dev/null 2>/dev/null

    elif [ -f "$PROFILE" ]; then
        # Subsequent iterations: use profile from previous run
        echo "  [1/4] Compiling (PGO=use, profile=$PROFILE)..."
        $CIRA -pass-pipeline="${PIPELINE},type2-gpu-codegen{pgo=use profile=$PROFILE})" \
            "$INPUT" -o /dev/null 2>/dev/null
    else
        echo "  [1/4] Compiling (PGO=instrument, no profile yet)..."
        $CIRA -pass-pipeline="${PIPELINE},type2-gpu-codegen{pgo=instrument})" \
            "$INPUT" -o /dev/null 2>/dev/null
    fi

    # Compile C++
    echo "  [2/4] Compiling C++..."
    g++ -O2 -std=c++17 -march=native -o /tmp/pgo_binary \
        type2_host_gpu.cpp "$TYPE2_SRC" \
        -I"$TYPE2_INC" -I"$CXLMEM_INC" 2>/dev/null

    # Execute and collect profile
    echo "  [3/4] Executing..."
    exec_output=$(timeout 30 /tmp/pgo_binary 2>&1)
    exec_time=$(echo "$exec_output" | grep "Completed in" | head -1 | grep -oP '\d+' || echo "?")
    echo "         Kernel time: ${exec_time}ms"

    if [ -f "$PROFILE" ]; then
        # Extract key metrics from profile
        tile=$(python3 -c "
import json
with open('$PROFILE') as f:
    data = json.load(f)
if data:
    d=data[0]
    print(f\"tile={d.get('tile_m',0)}x{d.get('tile_n',0)}x{d.get('tile_k',0)} pf={d.get('prefetch_dist',0)} hit={d.get('hit_rate',0):.2%}\")
    if 'suggested_tile_m' in d:
        print(f\"  → suggested: tile={d['suggested_tile_m']}x{d['suggested_tile_n']}x{d['suggested_tile_k']} pf={d['suggested_prefetch_dist']}\")
" 2>/dev/null || echo "no profile data")
        echo "  [4/4] Profile: $tile"

        # For next iteration: rewrite profile with suggested values
        if [ "$iter" -lt "$ITERS" ]; then
            python3 -c "
import json
with open('$PROFILE') as f:
    data = json.load(f)
for d in data:
    if 'suggested_tile_m' in d:
        d['tile_m'] = d['suggested_tile_m']
        d['tile_n'] = d['suggested_tile_n']
        d['tile_k'] = d['suggested_tile_k']
        d['prefetch_dist'] = d['suggested_prefetch_dist']
with open('$PROFILE', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null
        fi
    else
        echo "  [4/4] No profile generated (non-instrumented run)"
    fi
    echo ""
done

# Final optimized compilation (non-instrumented)
echo "━━━ Final optimized build ━━━"
if [ -f "$PROFILE" ]; then
    echo "  Compiling with PGO profile..."
    $CIRA -pass-pipeline="${PIPELINE},type2-gpu-codegen{pgo=use profile=$PROFILE})" \
        "$INPUT" -o /dev/null 2>/dev/null
else
    echo "  Compiling with default settings..."
    $CIRA -pass-pipeline="${PIPELINE},type2-gpu-codegen)" \
        "$INPUT" -o /dev/null 2>/dev/null
fi

g++ -O2 -std=c++17 -march=native -o /tmp/pgo_final \
    type2_host_gpu.cpp "$TYPE2_SRC" \
    -I"$TYPE2_INC" -I"$CXLMEM_INC" 2>/dev/null

echo "  Running final binary..."
final_output=$(timeout 30 /tmp/pgo_final 2>&1)
final_time=$(echo "$final_output" | grep "Completed in" | head -1 | grep -oP '\d+' || echo "?")
echo "  Final kernel time: ${final_time}ms"
echo ""

# Verify prefetch in generated code
pf_count=$(grep -c "__builtin_prefetch" type2_host_gpu.cpp || echo 0)
memcpy_count=$(grep -c "memcpy\b" type2_host_gpu.cpp || echo 0)
echo "═══════════════════════════════════════════════"
echo "  Prefetch instructions: $pf_count"
echo "  memcpy calls:          $memcpy_count (zero-copy)"
echo "  Final binary:          /tmp/pgo_final"
if [ -f "$PROFILE" ]; then
    echo "  Profile:               $PROFILE"
fi
echo "═══════════════════════════════════════════════"
