# MonetDB TPC-H gem5 RV64 Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile MonetDB through CXLMemUring, form and outline read-only memory-side regions, execute them on an instruction-retiring RV64 timing core in gem5-CXL, and prove Q1-Q22 correctness plus real device execution for Q1, Q6, Q9, Q11, and Q13 at SF=0.01.

**Architecture:** The x86 MonetDB process remains authoritative. Compiler-formed device regions receive only versioned shared-arena offsets, execute on a bare RV64 device context, and explicitly install host-consumed cache lines through the modeled CXL.cache path. A v2 kernel-ID job protocol connects generated x86 stubs, an RV64 firmware dispatch loop, and gem5's shared CXL arena. The simulator uses normal x86 SE for the host and a narrowly scoped physical-device mode for the RV64 timing CPU because gem5's `Root.full_system` mode is global.

**Tech Stack:** ClangIR/CIR, MLIR TableGen and C++ passes, LLVM IR/LLD, C/C++17 runtime, RISC-V cross GCC, MonetDB/MonetDBe, CMake/CTest/FileCheck, gem5 ALL-ISA SCons build, Python gem5 configuration, shell/Python result validation.

---

## Scope and execution rules

- The approved design contract is `docs/superpowers/specs/2026-08-31-monetdb-tpch-gem5-rv64-offload-design.md`.
- Work spans `/home/victoryang00/CXLMemUring` and `/home/victoryang00/gem5-CXL`; the MonetDB source is the nested repository at `bench/MonetDB`.
- Before implementation, invoke `superpowers:using-git-worktrees`. Create isolated worktrees/branches for both top-level repositories and, before changing MonetDB sources, for the nested MonetDB repository. Do not reset, clean, stash, or overwrite either current dirty checkout.
- Treat each task's test as a fail-closed gate. A native fallback for one of the required GDK units, an unknown kernel ID, a timeout, or a host-only target query is a failure even if SQL results are correct.
- Keep the device label in logs and reports exactly `gem5 RV64 Vortex surrogate`; do not call it cycle-accurate Vortex.
- Commit after every task. For a cross-repository task, commit the gem5 change in gem5-CXL first, then the matching CXLMemUring change. Never mix unrelated existing changes into these commits.

## Milestone 1: Freeze the ABI and compiler contracts

### Task 1: Establish isolated worktrees and baseline evidence

**Files:**

- Create: `/home/victoryang00/CXLMemUring-<worktree>/artifacts/baseline/cxlmemuring-status.txt`
- Create: `/home/victoryang00/CXLMemUring-<worktree>/artifacts/baseline/gem5-status.txt`
- Create: `/home/victoryang00/CXLMemUring-<worktree>/artifacts/baseline/compiler-tests.txt`
- Create: `/home/victoryang00/CXLMemUring-<worktree>/artifacts/baseline/runtime-tests.txt`

- [ ] **Step 1: Create isolated branches/worktrees**

Run the worktree skill, selecting branches such as `monetdb-rv64-offload` in CXLMemUring and `monetdb-rv64-device` in gem5-CXL. Initialize the MonetDB submodule in the CXLMemUring worktree, then create a MonetDB branch before Task 9.

Expected: the original paths still show the same dirty entries, while implementation paths are clean except for generated worktree metadata.

- [ ] **Step 2: Capture baseline identities without copying dirty state**

Run:

```bash
git -C /home/victoryang00/CXLMemUring status --short > artifacts/baseline/cxlmemuring-status.txt
git -C /home/victoryang00/gem5-CXL status --short > artifacts/baseline/gem5-status.txt
git rev-parse HEAD >> artifacts/baseline/cxlmemuring-status.txt
git -C /home/victoryang00/gem5-CXL rev-parse HEAD >> artifacts/baseline/gem5-status.txt
```

Expected: both files contain a commit ID and the pre-existing dirty-state inventory.

- [ ] **Step 3: Run the current focused tests**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j"$(nproc)" --target cira
ctest --test-dir build --output-on-failure -R 'region|offload|cache_completion' \
  | tee artifacts/baseline/compiler-tests.txt
ctest --test-dir build/runtime --output-on-failure \
  | tee artifacts/baseline/runtime-tests.txt
```

Expected: record every current pass or failure; do not repair unrelated baseline failures in this task.

- [ ] **Step 4: Commit only baseline manifests**

```bash
git add artifacts/baseline
git commit -m "test: capture MonetDB RV64 offload baselines"
```

### Task 2: Add a versioned kernel-ID CALL protocol

**Files:**

- Modify: `runtime/include/cira_cxl_job.h`
- Modify: `runtime/include/cira_mmio.h`
- Modify: `runtime/cira_mmio.cpp`
- Modify: `runtime/test/CMakeLists.txt`
- Create: `runtime/test/test_cira_kernel_call_abi.cpp`

- [ ] **Step 1: Write the failing ABI layout and publication-order test**

Add a runtime test which requires the old v1 payload to remain byte-for-byte available and a new v2 payload to use offsets and a stable ID:

```cpp
static_assert(sizeof(cira_cxl_call_job_v1_t) == 32);
static_assert(offsetof(cira_cxl_call_job_v2_t, kernel_id) == 0);
static_assert(offsetof(cira_cxl_call_job_v2_t, args_offset) == 8);
static_assert(offsetof(cira_cxl_call_job_v2_t, completion_offset) == 16);
static_assert(sizeof(cira_cxl_call_job_v2_t) == 40);

int main() {
    cira_mmio_window_t *window = nullptr;
    cira_mmio_config_t cfg{};
    cfg.emulate = true;
    assert(cira_mmio_open(&window, &cfg) == CIRA_MMIO_OK);
    uint64_t seq = 0;
    assert(cira_mmio_submit_kernel(window, 0x1122334455667788ULL,
                                   0x400, 96, 0x800, 0, &seq) == CIRA_MMIO_OK);
    auto *base = static_cast<const std::byte *>(cira_mmio_base(window));
    auto *slot = reinterpret_cast<const cira_cxl_arg_slot_t *>(
        base + cira_cxl_arg_slot_off(CIRA_CXL_JOB_CALL_V2));
    auto *job = reinterpret_cast<const cira_cxl_call_job_v2_t *>(slot + 1);
    assert(slot->version == CIRA_CXL_JOB_VERSION_KERNEL_ID);
    assert(job->kernel_id == 0x1122334455667788ULL);
    assert(job->args_offset == 0x400 && job->args_size == 96);
    assert(job->completion_offset == 0x800);
    assert(slot->seq == seq);
    cira_mmio_close(window);
}
```

- [ ] **Step 2: Verify it fails**

```bash
cmake --build build -j"$(nproc)" --target test_cira_kernel_call_abi
```

Expected: compilation fails because v2 types and `cira_mmio_submit_kernel` do not exist.

- [ ] **Step 3: Implement the additive ABI**

Keep `CIRA_CXL_JOB_CALL` and `cira_cxl_call_job_v1_t` for hardware compatibility. Add:

```c
#define CIRA_CXL_JOB_VERSION_KERNEL_ID 2u
#define CIRA_CXL_JOB_CALL_V2 5u

typedef struct {
    uint64_t kernel_id;
    uint64_t args_offset;
    uint64_t completion_offset;
    uint32_t args_size;
    uint32_t flags;
    uint64_t reserved;
} cira_cxl_call_job_v2_t;
```

Add `cira_mmio_submit_job_versioned(...)` internally so the slot header and doorbell both carry the requested version. Implement `cira_mmio_submit_kernel(...)` with nonzero ID, cache-line-aligned completion offset, maximum slot-size, and integer-overflow checks. Do not reinterpret a host function pointer as a kernel ID.

- [ ] **Step 4: Run ABI and existing MMIO tests**

```bash
cmake --build build -j"$(nproc)" --target test_cira_kernel_call_abi test_cira_offload_path
ctest --test-dir build/runtime --output-on-failure -R 'kernel_call_abi|offload_path|cache_completion'
```

Expected: all selected tests pass and the existing v1 test remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add runtime/include/cira_cxl_job.h runtime/include/cira_mmio.h \
  runtime/cira_mmio.cpp runtime/test
git commit -m "runtime: add versioned kernel-id call jobs"
```

### Task 3: Implement the bounded shared CXL arena

**Files:**

- Create: `runtime/include/cira_shared_arena.h`
- Create: `runtime/cira_shared_arena.cpp`
- Modify: `runtime/CMakeLists.txt`
- Modify: `runtime/test/CMakeLists.txt`
- Create: `runtime/test/test_cira_shared_arena.cpp`

- [ ] **Step 1: Write failing offset, alignment, and bounds tests**

Test this public C API:

```c
typedef struct cira_shared_arena cira_shared_arena_t;
int cira_shared_arena_open(cira_shared_arena_t **out, void *base,
                           uint64_t bytes, uint64_t control_bytes);
int cira_shared_arena_alloc(cira_shared_arena_t *arena, uint64_t bytes,
                            uint64_t alignment, uint64_t *out_offset);
int cira_shared_arena_to_offset(const cira_shared_arena_t *arena,
                                const void *ptr, uint64_t bytes,
                                uint64_t *out_offset);
void *cira_shared_arena_from_offset(const cira_shared_arena_t *arena,
                                    uint64_t offset, uint64_t bytes,
                                    uint64_t alignment);
```

The test must cover an 8 KiB reserved control prefix, 64-byte completion alignment, exact-end success, one-byte overflow failure, foreign-pointer failure, wraparound failure, and allocation exhaustion.

- [ ] **Step 2: Verify failure, then implement monotonically allocated arena state**

Use checked addition (`offset > size || bytes > size - offset`) everywhere. Protect host allocation with a mutex; keep translation functions allocation-free and usable by test firmware.

- [ ] **Step 3: Add runtime environment wiring**

Support `CIRA_CXL_ARENA_ADDR`, `CIRA_CXL_ARENA_SIZE`, and `CIRA_CXL_ARENA_CONTROL_BYTES`, with the control-byte default fixed to `CIRA_CXL_CONTROL_BYTES`. Expose the process-wide arena through `cira_shared_arena_default()` but require an explicit mapping in strict mode.

- [ ] **Step 4: Run tests**

```bash
cmake --build build -j"$(nproc)" --target test_cira_shared_arena
ctest --test-dir build/runtime --output-on-failure -R 'shared_arena|kernel_call_abi'
```

Expected: both tests pass under ASan in a Debug runtime build as well as RelWithDebInfo.

- [ ] **Step 5: Commit**

```bash
git add runtime/CMakeLists.txt runtime/include/cira_shared_arena.h \
  runtime/cira_shared_arena.cpp runtime/test
git commit -m "runtime: add bounded shared CXL arena"
```

### Task 4: Replace heuristic heap classification with explicit remote provenance

**Files:**

- Modify: `include/Dialect/Transforms/Passes.td`
- Modify: `include/Dialect/Transforms/BackwardSlice.h`
- Modify: `src/Dialect/Transforms/BackwardSlice.cpp`
- Modify: `src/Dialect/Transforms/RegionFormation.cpp`
- Modify: `test/CMakeLists.txt`
- Create: `test/region_provenance.mlir`

- [ ] **Step 1: Write negative and positive FileCheck cases**

Cover:

```mlir
!s64i = !cir.int<s, 64>

// CHECK-LABEL: @plain_heap
// CHECK-NOT: cira.offload
cir.func @plain_heap(%p: !cir.ptr<!s64i>) -> !s64i {
  %next = cir.load %p : !cir.ptr<!s64i>, !s64i
  %addr = cir.ptr_stride %p, %next : (!cir.ptr<!s64i>, !s64i) -> !cir.ptr<!s64i>
  %v = cir.load %addr : !cir.ptr<!s64i>, !s64i
  cir.return %v : !s64i
}

// CHECK-LABEL: @annotated_remote
// CHECK: cira.offload
func.func @annotated_remote(%p: memref<?xi64>, %i: index) -> i64 {
  %next = memref.load %p[%i] {cira.remote} : memref<?xi64>
  %next_i = arith.index_cast %next : i64 to index
  %v = memref.load %p[%next_i] {cira.remote} : memref<?xi64>
  return %v : i64
}

// CHECK-LABEL: @remote_memref
// CHECK: cira.offload
func.func @remote_memref(%p: memref<?xi64, 1>, %i: index) -> i64 {
  %next = memref.load %p[%i] : memref<?xi64, 1>
  %next_i = arith.index_cast %next : i64 to index
  %v = memref.load %p[%next_i] : memref<?xi64, 1>
  return %v : i64
}
```

Also test provenance propagation through `cir.cast`, `cir.ptr_stride`, `cir.get_member`, `llvm.getelementptr`, `memref.cast`, block arguments, and loop iter_args. A pointer returned by an unannotated call must be unknown, not remote.

- [ ] **Step 2: Confirm the current classifier fails the plain-heap case**

```bash
build/bin/cira test/region_provenance.mlir \
  --cira-region-formation='require-profitable=false' | FileCheck test/region_provenance.mlir
```

Expected: `plain_heap` is incorrectly outlined before the fix.

- [ ] **Step 3: Implement a tri-state provenance lattice**

Define `RemoteProvenance::{Local, Remote, Unknown}`. Seed Remote from CIRA handles, nonzero/symbolic memory spaces, `cira.remote` value/op attributes, and calls annotated `cira.remote_result`. Seed Local from allocas and `cira.local`. Propagate through pure pointer derivations; join conflicting or missing facts to Unknown. Only Remote roots are eligible unless `--allow-legacy-heap-remote=true` is explicitly selected.

- [ ] **Step 4: Run compiler tests**

```bash
cmake --build build -j"$(nproc)" --target cira
ctest --test-dir build --output-on-failure -R 'region_provenance|CXLMemUring_RunAll'
```

Expected: explicit cases pass and existing remote-address-space tests remain green.

- [ ] **Step 5: Commit**

```bash
git add include/Dialect/Transforms src/Dialect/Transforms test
git commit -m "compiler: require explicit remote provenance"
```

### Task 5: Merge compatible slices and separate submit from first dependent use

**Files:**

- Modify: `include/Dialect/Transforms/BackwardSlice.h`
- Modify: `src/Dialect/Transforms/BackwardSlice.cpp`
- Modify: `src/Dialect/Transforms/RegionFormation.cpp`
- Create: `test/region_merge_schedule.mlir`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Add failing merge and scheduling tests**

The test module must assert:

- two remote reads in one loop with overlapping address slices produce one `cira.offload`;
- a call or store between slices prevents merging;
- live-ins dominate the offload;
- `cira.barrier` appears immediately before the first host operation that consumes a prepared root, not automatically before every original load; and
- `cira.overlap_ops` is greater than zero for the compatible case.

Use checks such as:

```mlir
// CHECK-LABEL: func.func @merge_same_loop
// CHECK-COUNT-1: cira.offload
// CHECK: cira.overlap_ops = 2
// CHECK: arith.addi
// CHECK: cira.barrier
// CHECK: memref.load
```

- [ ] **Step 2: Verify it fails with one region per root**

```bash
build/bin/cira test/region_merge_schedule.mlir \
  --cira-region-formation='require-profitable=false' | FileCheck test/region_merge_schedule.mlir
```

- [ ] **Step 3: Implement candidate collection, compatibility, and placement**

Refactor `runOnOperation()` into three phases per function: collect `RegionCandidate`s without mutating IR, union compatible candidates within the same loop/control parent, then outline merged regions. Compatibility requires identical enclosing region, no intervening host-only effect, jointly device-legal ops, and dominance of all live-ins. Compute submit placement as the nearest common dominator after every live-in definition. Compute the wait point as the earliest host use of any prepared root result.

- [ ] **Step 4: Preserve read-only host authority**

Keep the original host operations. Reject device stores and yield no host-computed SQL values from the device region in this milestone. Add `cira.region_id`, `cira.slice_size`, `cira.root_count`, `cira.overlap_ops`, and source metadata to every accepted region.

- [ ] **Step 5: Run all formation tests and commit**

```bash
cmake --build build -j"$(nproc)" --target cira
ctest --test-dir build --output-on-failure -R 'region_(formation|provenance|merge_schedule)'
git add include/Dialect/Transforms src/Dialect/Transforms test
git commit -m "compiler: merge and schedule remote regions"
```

### Task 6: Outline typed RV64 kernels and emit a deterministic manifest

**Files:**

- Modify: `include/Dialect/Transforms/Passes.td`
- Modify: `include/Dialect/Transforms/Passes.h`
- Create: `src/Dialect/Transforms/OutlineRv64Kernels.cpp`
- Modify: `src/Dialect/Transforms/CMakeLists.txt`
- Create: `test/outline_rv64_kernels.mlir`
- Create: `test/check_kernel_manifest.py`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing outline and determinism tests**

Require the pass interface:

```text
--cira-outline-rv64-kernels='manifest=/tmp/kernels.json module-id=monetdb'
```

For a region with two live-ins, FileCheck must see a host region annotated with `cira.kernel_id`, a generated `func.func private @__cira_rv64_<16-hex-digits>(%arena: i64, %args_offset: i64)`, and a packed argument-layout attribute. Running the pass twice must produce byte-identical JSON.

- [ ] **Step 2: Define stable identity and manifest schema**

Use SHA-256 over the canonical tuple `module-id | normalized source path | function symbol | source line:column | region ordinal`, truncate to the first 64 bits, and fail on collision within a module. Emit sorted JSON records:

```json
{
  "abi_version": 2,
  "kernels": [{
    "kernel_id": "0x0123456789abcdef",
    "symbol": "__cira_rv64_0123456789abcdef",
    "source": "gdk/gdk_join.c:4451:1",
    "query_tags": [9, 11],
    "args_size": 32,
    "args": [{"name":"link_base","offset":0,"type":"arena_offset"}],
    "read_only": true
  }]
}
```

- [ ] **Step 3: Implement typed packing and legality diagnostics**

Convert remote pointers to `i64` arena offsets in the device signature. Preserve integers/floats by value. Reject host pointers, function pointers, unresolved calls, allocation, stores, atomics, and unsupported types with an error naming the source operation and boundary reason. Clone body block arguments through an `IRMapping`; this replaces the current zero-argument clone in `ConvertCiraToLLVMHeteroPass`.

- [ ] **Step 4: Test deterministic output and rejection**

```bash
cmake --build build -j"$(nproc)" --target cira
ctest --test-dir build --output-on-failure -R 'outline_rv64|kernel_manifest'
```

Expected: deterministic positive case passes; an unresolved host call is a `WILL_FAIL` CTest with a stable diagnostic.

- [ ] **Step 5: Commit**

```bash
git add include/Dialect/Transforms src/Dialect/Transforms test
git commit -m "compiler: outline deterministic RV64 kernels"
```

### Task 7: Lower host submissions to arena offsets and kernel IDs

**Files:**

- Modify: `src/Conversion/CiraToLLVM.cpp`
- Modify: `runtime/src/CiraRuntime.cpp`
- Modify: `runtime/cira_jit_engine.cpp`
- Create: `test/hetero_kernel_submit.mlir`
- Modify: `test/CMakeLists.txt`
- Modify: `runtime/test/test_cira_offload_path.cpp`

- [ ] **Step 1: Write a failing lowering test**

FileCheck must require calls with values, not x86 function addresses:

```mlir
// CHECK: llvm.mlir.constant(81985529216486895 : i64)
// CHECK: llvm.call @cira_offload_submit_kernel
// CHECK-NOT: llvm.call @cira_get_device_func_addr
// CHECK: llvm.load
// CHECK: llvm.call @cira_future_await
```

The runtime test must inspect the emulated window and prove that the submitted payload contains the manifest ID and arena-relative argument/completion offsets.

- [ ] **Step 2: Add the runtime API**

Expose:

```c
int cira_offload_submit_kernel(uint64_t kernel_id,
                               const void *typed_args, uint32_t args_size,
                               void *completion, uint32_t flags);
```

The function allocates/copies the typed argument block in the shared arena, translates completion to an offset, submits v2, tracks the sequence, and returns a status enum. In strict gem5 mode, missing arena/MMIO, conversion failure, or submit failure is recorded and returned; never silently fabricate completion.

- [ ] **Step 3: Replace heterogeneous lowering**

Read `cira.kernel_id` and the packed-layout metadata from the outlined region, materialize the typed argument struct, and call the new API. Clone the original region body on x86 exactly once. Emit the existing future wait only at the barrier chosen by Task 5.

- [ ] **Step 4: Run lowering/runtime tests and commit**

```bash
cmake --build build -j"$(nproc)" --target cira test_cira_offload_path
ctest --test-dir build --output-on-failure -R 'hetero_kernel_submit|offload_path|kernel_call_abi'
git add src/Conversion runtime test
git commit -m "compiler: lower offloads to kernel-id submissions"
```

## Milestone 2: Produce an independent RV64 device image

### Task 8: Build the no-libc RV64 firmware, dispatch table, and device helpers

**Files:**

- Create: `runtime/device/cira_rv64_start.S`
- Create: `runtime/device/cira_rv64_firmware.c`
- Create: `runtime/device/cira_rv64_device.h`
- Create: `runtime/device/cira_rv64_link.ld`
- Create: `runtime/device/CMakeLists.txt`
- Create: `runtime/device/gen_dispatch_table.py`
- Modify: `runtime/CMakeLists.txt`
- Create: `runtime/test/test_rv64_dispatch_table.py`
- Modify: `runtime/test/CMakeLists.txt`

- [ ] **Step 1: Write the dispatch-table generator test**

Given a two-kernel manifest, require sorted output:

```c
static const struct cira_kernel_entry cira_kernel_table[] = {
    { UINT64_C(0x0000000000000002), __cira_rv64_0000000000000002, 32 },
    { UINT64_C(0x0000000000000009), __cira_rv64_0000000000000009, 48 },
};
```

Duplicate IDs, an args-size mismatch, or a non-read-only kernel must make the generator exit nonzero.

- [ ] **Step 2: Implement firmware validation and publication order**

The firmware loop must:

```c
for (;;) {
    uint64_t seq = cira_load_acquire64(&doorbell->seq);
    if (seq == last_seq) { cira_device_pause(); continue; }
    struct cira_cxl_call_job_v2 job;
    if (!cira_snapshot_and_validate(seq, &job)) {
        cira_publish_error(seq, CIRA_CXL_STATUS_BAD_DESCRIPTOR);
        last_seq = seq;
        continue;
    }
    const struct cira_kernel_entry *entry = cira_find_kernel(job.kernel_id);
    if (!entry || entry->args_size != job.args_size) {
        cira_publish_error(seq, CIRA_CXL_STATUS_UNKNOWN_KERNEL);
        last_seq = seq;
        continue;
    }
    entry->fn(CIRA_DEVICE_ARENA_BASE, job.args_offset);
    cira_publish_success(seq, job.completion_offset);
    last_seq = seq;
}
```

All addresses are checked offsets. `cira_publish_success` writes result/status first, issues a release fence, and writes completion sequence/magic last.

- [ ] **Step 3: Add explicit device cache-install helper**

Define a custom RV64 pseudo instruction wrapper `cira_device_install_cacheline(offset)` reserved for gem5. Ordinary RV64 loads must remain ordinary loads. The outlined kernel lowering inserts this helper only for prepared root lines.

- [ ] **Step 4: Build and inspect the ELF**

```bash
cmake -S runtime/device -B build/rv64-device \
  -DCMAKE_C_COMPILER=/usr/bin/riscv64-linux-gnu-gcc \
  -DCIRA_KERNEL_MANIFEST="$PWD/build/monetdb/kernels.json"
cmake --build build/rv64-device -j"$(nproc)"
riscv64-linux-gnu-readelf -hWs build/rv64-device/cira-rv64-device.elf
riscv64-linux-gnu-nm -u build/rv64-device/cira-rv64-device.elf
```

Expected: ELF64 RISC-V, entry `_start`, firmware and kernel symbols present, and no undefined symbols.

- [ ] **Step 5: Run generator tests and commit**

```bash
ctest --test-dir build/runtime --output-on-failure -R 'rv64_dispatch'
git add runtime/device runtime/CMakeLists.txt runtime/test
git commit -m "runtime: add RV64 firmware and kernel dispatch"
```

## Milestone 3: Compile and drive real MonetDB TPC-H

### Task 9: Make the MonetDB build targeted, reproducible, and fail closed

**Files:**

- Modify: `scripts/build.sh`
- Create: `scripts/monetdb_cira_units.txt`
- Create: `scripts/build_monetdb_tpch_cira.sh`
- Create: `scripts/smoke_mserver5.py`
- Modify in MonetDB: `bench/MonetDB/gdk/gdk_utils.h`
- Modify in MonetDB: `bench/MonetDB/gdk/gdk_utils.c`
- Modify in MonetDB: `bench/MonetDB/gdk/gdk_storage.c`
- Create: `test/test_monetdb_build_contract.sh`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write a failing build-contract test**

Require the targeted list to include at least:

```text
gdk/gdk_select.c q6 scan
gdk/gdk_aggr.c q1 aggregation
gdk/gdk_join.c q9,q11 join
gdk/gdk_cand.c q11 candidate
gdk/gdk_string.c q13 grouping-traversal
monetdb5/modules/mal/pcre.c q13 string-group
```

The test must fail if `scripts/build.sh` sends any listed file through the native `gdk_*.c` fallback or omits `--cira-region-formation` and `--cira-outline-rv64-kernels` in heterogeneous mode. Add a translation-readiness subtest which runs every listed unit through the frontend and pass pipeline independently. The known `gdk_aggr.c` deep-CFG/printer failure is part of this gate: if it recurs, stop and use `superpowers:systematic-debugging`, reduce it to a checked-in compiler regression, fix the compiler/pipeline scalability defect, and rerun all listed units; do not bypass the unit or hand-write its kernel.

- [ ] **Step 2: Add an explicit MonetDB CXL-allocation hook**

Under `#ifdef CIRA_MONETDB_SHARED_ARENA`, route BAT heap mappings used by the SF=0.01 runner through `cira_shared_arena_alloc`, mark returned storage with `cira.remote`, and preserve existing allocation/free behavior when the macro is absent. Do not place locks, catalogs, parser state, or arbitrary GDK control allocations in the arena.

Expose a small hook in `gdk_utils.h`:

```c
typedef void *(*gdk_cira_alloc_fn)(size_t bytes, size_t alignment);
typedef void (*gdk_cira_free_fn)(void *ptr, size_t bytes);
gdk_export void GDKset_cira_heap_hooks(gdk_cira_alloc_fn alloc_fn,
                                      gdk_cira_free_fn free_fn);
```

Only `HEAPalloc`/`HEAPextend` storage for BAT tail/value/hash arrays uses the hook.

- [ ] **Step 3: Refactor the build mode**

Make `PIPELINE_LLVM_LOWERING=hetero` select this pass chain for listed units:

```text
cir-mlir-scf-prepare,cir-to-mlir,
cir-to-cira,
cira-region-formation{require-profitable=<mode>},
cira-outline-rv64-kernels{manifest=<path> module-id=monetdb},
convert-cira-to-llvm-hetero,
convert-func-to-llvm,reconcile-unrealized-casts
```

Unlisted problematic units may remain native. A listed unit may not fall back. Produce `build/monetdb_pipeline/build-contract.json` with source, compiler path, pass list, object path, and kernel IDs.

- [ ] **Step 4: Build the full server and smoke it**

```bash
PIPELINE_LLVM_LOWERING=hetero CIRA_OFFLOAD_MODE=forced-debug \
  scripts/build_monetdb_tpch_cira.sh --server-only
bin/monetdb/x86_64-unknown-linux-gnu/mserver5 --version
scripts/smoke_mserver5.py \
  --server bin/monetdb/x86_64-unknown-linux-gnu/mserver5 \
  --dbpath artifacts/tpch/mserver-smoke
```

Expected: a current `mserver5` is linked; build-contract JSON lists every required unit as `cira-hetero`; `--version` exits zero; the smoke helper observes a live server, performs a clean shutdown, and records stdout/stderr without deleting a pre-existing database path.

- [ ] **Step 5: Commit nested MonetDB, then the superproject**

```bash
git -C bench/MonetDB add gdk/gdk_utils.h gdk/gdk_utils.c gdk/gdk_storage.c
git -C bench/MonetDB commit -m "gdk: add opt-in CIRA BAT heap hooks"
git add bench/MonetDB scripts test
git commit -m "build: compile targeted MonetDB units through CIRA"
```

### Task 10: Add the single-process TPC-H runner and exact result oracle

**Files:**

- Create: `bench/tpch_runner/CMakeLists.txt`
- Create: `bench/tpch_runner/monetdb_tpch_cira.c`
- Create: `bench/tpch_runner/result_writer.c`
- Create: `bench/tpch_runner/result_writer.h`
- Create: `scripts/run_tpch_sf001.py`
- Create: `scripts/compare_tpch_results.py`
- Create: `test/test_tpch_result_compare.py`
- Modify: `scripts/build_monetdb_tpch_cira.sh`

- [ ] **Step 1: Write result-normalization tests first**

Use fixtures with NULL, DECIMAL, DATE, strings containing separators, ordered and unordered queries, and one-ULP numeric differences. The comparator must compare MonetDB's rendered typed values exactly after canonical formatting; it must not apply a blanket floating tolerance. Sort rows only for queries whose SQL has no top-level `ORDER BY`.

- [ ] **Step 2: Implement the MonetDBe runner**

The binary accepts:

```text
monetdb-tpch-cira --data <SF-0.01> --queries <queries> --output <dir>
                  --mode baseline|forced-debug --query 1|all
```

It opens one embedded database, runs `schema.sql`, performs `COPY` for the eight checked-in SF=0.01 tables, verifies counts `5,25,100,1500,2000,8000,15000,70000`, then runs q01-q22. Emit one typed TSV plus JSON metadata per query. Surround each query with `m5_work_begin(query_id, mode)` / `m5_work_end(query_id, mode)` when built for gem5.

- [ ] **Step 3: Attribute required query regions**

Set query context in the runtime before executing each SQL file. The runtime writes the active query ID into submit telemetry, allowing the generic compiler mechanism to prove Q1/Q6/Q9/Q11/Q13 coverage without hard-coding SQL numbers into region formation.

- [ ] **Step 4: Build and run native baseline/offload correctness**

```bash
scripts/build_monetdb_tpch_cira.sh --runner
scripts/run_tpch_sf001.py --binary bin/monetdb/x86_64-unknown-linux-gnu/monetdb-tpch-cira \
  --mode baseline --output artifacts/tpch/native-baseline
scripts/run_tpch_sf001.py --binary bin/monetdb/x86_64-unknown-linux-gnu/monetdb-tpch-cira \
  --mode forced-debug --output artifacts/tpch/native-offload
scripts/compare_tpch_results.py artifacts/tpch/native-baseline artifacts/tpch/native-offload
```

Expected: `Q01..Q22 PASS`; native offload may report `no-device` but must preserve results. This step does not satisfy device coverage.

- [ ] **Step 5: Commit**

```bash
git add bench/tpch_runner scripts test
git commit -m "bench: add in-process MonetDB TPC-H runner"
```

## Milestone 4: Add the RV64 device context and shared CXL path to gem5

### Task 11: Add a narrowly scoped physical-device mode to gem5 CPUs

**Repository:** `/home/victoryang00/gem5-CXL`

**Files:**

- Modify: `src/cpu/BaseCPU.py`
- Modify: `src/cpu/base.hh`
- Modify: `src/cpu/base.cc`
- Modify: `src/cpu/simple/base.cc`
- Modify: `src/arch/riscv/tlb.cc`
- Create: `tests/gem5/cira_rv64_device/configs/device_physical_smoke.py`
- Create: `tests/gem5/cira_rv64_device/src/spin.S`
- Create: `tests/gem5/cira_rv64_device/test_device_physical.py`

- [ ] **Step 1: Write a smoke configuration that currently fails**

Construct an x86 SE process and a `RiscvTimingSimpleCPU` with no `Process`, `device_physical_mode=True`, and a tiny RV64 ELF that increments a counter at a physical address and spins. The initial expected failure is BaseCPU's `cpu.workload` count check.

- [ ] **Step 2: Add the guarded CPU parameter**

Add `device_physical_mode = Param.Bool(False, ...)`. When false, preserve all existing checks. When true, require exactly one thread, an empty workload vector, RISC-V ISA, and a configured device loader. Add `BaseCPU::isDevicePhysicalMode()`.

- [ ] **Step 3: Make SimpleCPU construct a process-less thread only in device mode**

Use the existing full-system-style `SimpleThread(..., System*, MMU*, ISA*, Decoder*)` constructor for that CPU while leaving global `FullSystem=false`. In the RISC-V TLB, if `tc->getCpuPtr()->isDevicePhysicalMode()`, validate the address with PMA, set `Request::PHYSICAL`, and map virtual to identical physical for atomic, timing, and functional translations. Do not change x86 or ordinary RISC-V SE translation.

- [ ] **Step 4: Run the focused test and SE regressions**

```bash
scons build/ALL/gem5.opt -j"$(nproc)"
build/ALL/gem5.opt tests/gem5/cira_rv64_device/configs/device_physical_smoke.py
./main.py run gem5/se_mode/hello_se --length=quick
```

Expected: the smoke prints `RV64_DEVICE_COUNTER=1`; x86 and RISC-V SE hello tests remain passing.

- [ ] **Step 5: Commit**

```bash
git add src/cpu src/arch/riscv/tlb.cc tests/gem5/cira_rv64_device
git commit -m "cpu: support process-less RV64 device contexts"
```

### Task 12: Implement the RV64 ELF loader and shared-arena registration

**Repository:** `/home/victoryang00/gem5-CXL`

**Files:**

- Create: `src/dev/cira/CiraRv64Device.py`
- Create: `src/dev/cira/cira_rv64_device.hh`
- Create: `src/dev/cira/cira_rv64_device.cc`
- Create: `src/dev/cira/SConscript`
- Modify: `src/SConscript`
- Modify: `include/gem5/asm/generic/m5ops.h`
- Modify: `include/gem5/m5ops.h`
- Modify: `src/sim/pseudo_inst.hh`
- Modify: `src/sim/pseudo_inst.cc`
- Modify: `util/m5/src/abi/x86/m5op.S`
- Create: `tests/gem5/cira_rv64_device/src/arena_roundtrip.c`
- Modify: `tests/gem5/cira_rv64_device/configs/device_physical_smoke.py`

- [ ] **Step 1: Add a failing arena round-trip test**

The x86 host reserves a page-aligned but untouched virtual range, calls `m5_cira_register_arena(base, size)`, writes a command at offset `0x2000`, and waits. RV64 reads device aperture `CIRA_DEVICE_ARENA_BASE + 0x2000`, writes a completion at `+0x2040`, and spins. The test must fail if the virtual range is already mapped, the configured CXL pool cannot provide one contiguous extent, or the extent is not visible to the device.

- [ ] **Step 2: Implement registration with a fail-closed physical contract**

Add one x86 m5op:

```c
uint64_t m5_cira_register_arena(const void *base, uint64_t size);
```

Add `cxl_mem_pool_id` to `CiraRv64Device`. On registration, require an unmapped page-aligned x86 SE virtual reservation, allocate one contiguous extent with `SEWorkload::allocPhysPages(npages, cxl_mem_pool_id)`, require that extent to equal the configured CXL range/pool, map it into the calling `Process` with `Process::map`, and record `{host_vbase, phys_base, size}`. Return the device aperture base on success and zero on failure. This makes offsets portable, guarantees that arena pages route through the CXL memory controller, and avoids both x86 virtual pointers and leaked replaced mappings.

- [ ] **Step 3: Implement the loader lifecycle**

`CiraRv64Device` receives the RV64 CPU, ELF path, instruction-memory range, stack top, CXL physical range, aperture base, and control-window offset. In `initState()` it uses `loader::createObjectFile`, rejects a non-RISC-V ELF or overlapping segments, writes segments through `system->physProxy`, zeroes BSS, initializes PC/SP/A0-A2, and activates only the device thread. Export kernel symbol ranges for stats.

- [ ] **Step 4: Verify x86-to-RV64 round trip**

```bash
scons build/ALL/gem5.opt -j"$(nproc)"
build/ALL/gem5.opt tests/gem5/cira_rv64_device/configs/device_physical_smoke.py \
  --test arena-roundtrip
```

Expected: output includes `ARENA_REGISTERED`, `RV64_READ_OK`, and `RV64_WRITE_OK`; no functional memory shortcut is used after initialization.

- [ ] **Step 5: Commit**

```bash
git add src/dev/cira src/SConscript include/gem5 src/sim util/m5 \
  tests/gem5/cira_rv64_device
git commit -m "dev: load RV64 firmware into a shared CXL arena"
```

### Task 13: Connect RV64 traffic, explicit cache installs, and proof statistics

**Repository:** `/home/victoryang00/gem5-CXL`

**Files:**

- Modify: `src/dev/cira/CiraRv64Device.py`
- Modify: `src/dev/cira/cira_rv64_device.hh`
- Modify: `src/dev/cira/cira_rv64_device.cc`
- Modify: `src/mem/CIRA.py`
- Modify: `src/mem/cira.hh`
- Modify: `src/mem/cira.cc`
- Modify: `include/gem5/asm/generic/m5ops.h`
- Modify: `include/gem5/m5ops.h`
- Modify: `src/sim/pseudo_inst.hh`
- Modify: `src/sim/pseudo_inst.cc`
- Modify: `util/m5/src/abi/riscv/m5op.S`
- Create: `tests/gem5/cira_rv64_device/src/install_cacheline.S`
- Create: `tests/gem5/cira_rv64_device/test_device_stats.py`

- [ ] **Step 1: Write a failing distinction test**

Run one RV64 ordinary load and one `cira_device_install_cacheline`. Require:

```text
system.cira_rv64_device.rv64Insts > 0
system.cira_rv64_device.cxlReadPackets > 0
system.cira.issuedDeviceInstalls == 1
system.cira.issuedPrefetches == 1
```

The ordinary load may increment RV64/CXL read stats, but must not increment `issuedDeviceInstalls`.

- [ ] **Step 2: Add the reserved RV64 instruction**

Add `M5OP_CIRA_DEVICE_INSTALL` to the shared m5op table and RISC-V libm5 wrapper. Its pseudo-instruction action calls `CiraRv64Device::installHostCacheline(tc, arena_offset)`, validates the offset, translates to CXL physical address, and submits one request through CIRA's timing port toward the host cache hierarchy. Illegal offsets return zero, mark the current job failed, and increment `badArenaOffsets`; valid requests return a nonzero request ID.

- [ ] **Step 3: Add per-kernel and protocol stats**

Record submitted/completed/failed/pending jobs, last submitted/completed sequence, unknown IDs, bad versions/bounds, RV64 retired instructions, instructions within each manifest symbol range, ordinary CXL reads/bytes, explicit installs, job latency, and submit-to-first-host-use overlap ticks. Expose a machine-readable summary at simulation exit.

- [ ] **Step 4: Run focused test and commit**

```bash
scons build/ALL/gem5.opt -j"$(nproc)"
python3 tests/gem5/cira_rv64_device/test_device_stats.py \
  --gem5 build/ALL/gem5.opt
git add src/dev/cira src/mem src/arch/riscv tests/gem5/cira_rv64_device
git commit -m "mem: account RV64 CXL traffic and cache installs"
```

### Task 14: Add malformed-job and firmware execution integration tests

**Repositories:** gem5-CXL and CXLMemUring

**Files in gem5-CXL:**

- Create: `configs/example/gem5_library/x86-monetdb-rv64-cxl-se.py`
- Create: `tests/gem5/cira_rv64_device/configs/firmware_job_smoke.py`
- Create: `tests/gem5/cira_rv64_device/test_firmware_jobs.py`

**Files in CXLMemUring:**

- Create: `runtime/device/test_kernels.c`
- Create: `scripts/build_rv64_smoke_image.sh`

- [ ] **Step 1: Add the complete heterogeneous gem5 config**

Instantiate one x86 timing/O3 host CPU, one `RiscvTimingSimpleCPU(device_physical_mode=True)`, separate host-local and device-local ranges, one shared CXL range through `SerialLink`, CIRA attached toward the host cache hierarchy, and `CiraRv64Device`. Add arguments for both ELFs, CPU types/clocks, CXL delay/speed/lanes, arena size, instruction range, and output summary.

- [ ] **Step 2: Build a synthetic image with one known kernel**

The kernel reads an array from the arena, issues cache installs for each line, and writes no SQL-visible data. Build it with the real v2 headers, manifest generator, firmware, and linker script.

- [ ] **Step 3: Exercise success and every fail-closed case**

The test matrix is: valid job, unknown ID, bad ABI version, stale sequence, args overflow, misaligned completion, missing completion, and kernel illegal instruction. Each case checks exact status and ensures the firmware returns to the service loop except for the intentional device-fault case.

- [ ] **Step 4: Run and verify evidence**

```bash
scripts/build_rv64_smoke_image.sh
/home/victoryang00/gem5-CXL/build/ALL/gem5.opt \
  /home/victoryang00/gem5-CXL/tests/gem5/cira_rv64_device/configs/firmware_job_smoke.py \
  --host-bin build/runtime/test/cira-job-host \
  --device-elf build/rv64-device/cira-rv64-smoke.elf
python3 /home/victoryang00/gem5-CXL/tests/gem5/cira_rv64_device/test_firmware_jobs.py \
  --m5out m5out
```

Expected: valid job has matching nonzero sequences, RV64 symbol-range instructions, CXL reads, installs, and completion; malformed cases report their specific terminal status with zero false success.

- [ ] **Step 5: Commit both sides**

Commit CXLMemUring as `test: add RV64 firmware job smoke image`, then gem5-CXL as `test: cover heterogeneous CIRA firmware jobs`.

## Milestone 5: End-to-end SF=0.01 validation

### Task 15: Build all artifacts and prove native/server gates

**Files:**

- Modify: `scripts/build_monetdb_tpch_cira.sh`
- Create: `scripts/check_monetdb_artifacts.py`
- Create: `artifacts/tpch/.gitkeep`

- [ ] **Step 1: Add an artifact contract checker**

Require fresh mtimes and hashes for `mserver5`, `monetdb-tpch-cira`, host LLVM IR/object inventory, RV64 kernel LLVM IR/objects, `kernels.json`, dispatch table, device ELF, and `build/ALL/gem5.opt`. Verify every manifest symbol appears in the ELF and every required GDK unit appears in build-contract JSON as CIRA-compiled.

- [ ] **Step 2: Perform a clean isolated build**

Remove only task-owned build directories inside the isolated worktrees, then run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j"$(nproc)"
PIPELINE_LLVM_LOWERING=hetero CIRA_OFFLOAD_MODE=forced-debug \
  scripts/build_monetdb_tpch_cira.sh --all
scons -C /home/victoryang00/gem5-CXL build/ALL/gem5.opt -j"$(nproc)"
scripts/check_monetdb_artifacts.py --cxlmemuring . \
  --gem5 /home/victoryang00/gem5-CXL
```

Expected: checker prints `ARTIFACT_CONTRACT PASS` and names all five required query tags.

- [ ] **Step 3: Run compiler/runtime/server tests**

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build/runtime --output-on-failure
bin/monetdb/x86_64-unknown-linux-gnu/mserver5 --version
```

Expected: all task-added tests pass and server version exits zero.

- [ ] **Step 4: Commit build orchestration**

```bash
git add scripts artifacts/tpch/.gitkeep
git commit -m "build: verify MonetDB heterogeneous artifacts"
```

### Task 16: Run matched gem5 baseline and offload modes for Q1-Q22

**Files:**

- Create: `scripts/run_monetdb_tpch_gem5.py`
- Create: `scripts/check_gem5_tpch_run.py`
- Create: `test/test_check_gem5_tpch_run.py`

- [ ] **Step 1: Write verifier fixture tests**

Fixtures must prove the verifier rejects: one result mismatch, one missing target submit, zero RV64 instructions in the expected symbol range, zero installs, sequence mismatch, pending job, config mismatch, fallback-only target, and an unlabeled device. A fully populated fixture must pass.

- [ ] **Step 2: Implement matched-run orchestration**

For each query, run baseline and forced-debug with identical x86 CPU, clock, caches, memory, SerialLink, dataset, and query input. Only offload enablement and RV64 activity differ. Put each run under `artifacts/tpch/gem5/{baseline,offload}/qNN/`; never reuse `m5out` across queries.

- [ ] **Step 3: Execute Q1-Q22**

```bash
scripts/run_monetdb_tpch_gem5.py \
  --gem5 /home/victoryang00/gem5-CXL/build/ALL/gem5.opt \
  --config /home/victoryang00/gem5-CXL/configs/example/gem5_library/x86-monetdb-rv64-cxl-se.py \
  --host-bin bin/monetdb/x86_64-unknown-linux-gnu/monetdb-tpch-cira \
  --device-elf build/rv64-device/cira-rv64-device.elf \
  --sf 0.01 --queries 1-22 --cpu timing --cxl-delay 1us
```

Expected: every run exits normally and creates `config.ini`, `stats.txt`, stdout/stderr, typed result, telemetry JSON, and manifest snapshot.

- [ ] **Step 4: Enforce correctness and five-query execution gates**

```bash
scripts/check_gem5_tpch_run.py \
  --root artifacts/tpch/gem5 \
  --manifest build/monetdb_pipeline/kernels.json \
  --required-offload 1,6,9,11,13
```

Expected final lines:

```text
TPC-H Q01-Q22 RESULT_EQUIVALENCE PASS
TPC-H Q01,Q06,Q09,Q11,Q13 DEVICE_EXECUTION PASS
DEVICE_MODEL gem5 RV64 Vortex surrogate
PERFORMANCE_GATE not-applied SF=0.01 forced-debug
```

- [ ] **Step 5: Commit the reusable harness, not bulky run outputs**

```bash
git add scripts/run_monetdb_tpch_gem5.py scripts/check_gem5_tpch_run.py \
  test/test_check_gem5_tpch_run.py
git commit -m "test: validate TPC-H RV64 offload in gem5"
```

### Task 17: Produce the final evidence report and perform completion review

**Files:**

- Create: `docs/results/2026-08-31-monetdb-tpch-gem5-rv64-sf001.md`
- Create: `artifacts/tpch/gem5/summary.json`
- Modify: `README.md`

- [ ] **Step 1: Generate a result table from artifacts**

For Q1-Q22 report result digest equality, row count, simulated ROI ticks, host instructions, and status. For Q1/Q6/Q9/Q11/Q13 additionally report kernel IDs/symbols, submitted/completed/failed/pending, sequence equality, RV64 symbol-range instructions, CXL reads/bytes, cache installs, job latency, and overlap ticks.

- [ ] **Step 2: Check topology evidence directly**

Quote the relevant `config.ini` values for x86 CPU type, RV64 CPU type and `device_physical_mode`, clocks, memory ranges, SerialLink delay/speed/lanes, CIRA connection, arena range, and both binaries. The report must distinguish device-local instruction memory from shared CXL memory.

- [ ] **Step 3: State proof boundaries**

State that SF=0.01 forced-debug proves correctness and execution coverage only. State that no speedup threshold was applied. Label the device as a gem5 RV64 Vortex surrogate and reserve profitability/performance claims for a separately approved SF=1 measured campaign.

- [ ] **Step 4: Invoke verification-before-completion and requesting-code-review**

Run every command in the final verification block below, inspect outputs, then invoke `superpowers:requesting-code-review`. Resolve substantive review findings and rerun affected tests before claiming completion.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/results/2026-08-31-monetdb-tpch-gem5-rv64-sf001.md \
  artifacts/tpch/gem5/summary.json
git commit -m "docs: report MonetDB TPC-H RV64 gem5 proof"
```

## Final verification block

Run from the isolated CXLMemUring worktree after all tasks:

```bash
git status --short
git -C /home/victoryang00/gem5-CXL status --short
git -C bench/MonetDB status --short

cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
ctest --test-dir build/runtime --output-on-failure

scripts/check_monetdb_artifacts.py --cxlmemuring . \
  --gem5 /home/victoryang00/gem5-CXL

/home/victoryang00/gem5-CXL/build/ALL/gem5.opt \
  /home/victoryang00/gem5-CXL/tests/gem5/cira_rv64_device/configs/firmware_job_smoke.py \
  --host-bin build/runtime/test/cira-job-host \
  --device-elf build/rv64-device/cira-rv64-smoke.elf

scripts/check_gem5_tpch_run.py \
  --root artifacts/tpch/gem5 \
  --manifest build/monetdb_pipeline/kernels.json \
  --required-offload 1,6,9,11,13
```

Required outcome:

- no unexpected dirty files in any isolated implementation worktree;
- all compiler and runtime tests pass;
- the synthetic firmware test proves real RV64 retirement, CXL reads, explicit installs, and completion;
- Q1-Q22 results match exactly at SF=0.01;
- Q1/Q6/Q9/Q11/Q13 meet every device-execution gate;
- no failed or pending job remains; and
- the current dirty state of the original user worktrees is unchanged.
