# MonetDB TPC-H Generic RV64 Offload in gem5-CXL

Date: 2026-08-31

## Summary

CXLMemUring will compile MonetDB and form generic device kernels from
memory-side dependence slices in selected TPC-H query paths. The x86 host will
run the MonetDB query engine while a separate simulated RV64 timing core runs a
bare-metal CIRA firmware loop and the outlined kernels. Both processors will
share a modeled CXL-memory arena and communicate through the existing CIRA job
ABI.

The first validation point is gem5-CXL at TPC-H SF=0.01. All Q1-Q22 must match
the baseline query results. Q1, Q6, Q9, Q11, and Q13 must additionally prove
that a job was submitted, that RV64 instructions executed in the corresponding
device kernel, that the device generated CXL traffic and coherent cache-line
installations, and that completion reached the host without a failed or
stranded job.

This checkout does not contain a Vortex SIMT CPU model. The device will
therefore use gem5's RV64 timing CPU as an instruction-executing
Vortex-compatible surrogate. Results must be labeled accordingly and must not
be presented as cycle-accurate Vortex timing.

## Current-State Problems

The existing MonetDB build path does not exercise the CIRA compiler:

- `scripts/build.sh monetdb` selects the direct CIR-to-LLVM path by default.
- `gdk_*.c`, including the join and aggregation implementation files, is on the
  native-compilation fallback list.
- The build script does not run `cira-region-formation` before heterogeneous
  lowering.
- No current `build/monetdb_pipeline` artifacts exist; the existing
  `bin/monetdb/x86_64-unknown-linux-gnu/mserver5` predates the current compiler
  and runtime changes.
- The existing region formation creates one candidate per remote access, does
  not merge overlapping loop slices, and places synchronization too close to
  the root access to guarantee useful overlap.
- The existing heterogeneous lowering uses a host function pointer for a
  generic `CALL`; an x86 pointer is not a valid RV64 device entry point.
- gem5-CXL models CIRA prefetch requests but has no configuration that runs an
  x86 MonetDB host and an instruction-executing RV64 device together.

These gaps mean that a successful `mserver5` link or a nonzero gem5 CIRA
counter is not sufficient evidence of compiler-formed device execution.

## Goals

1. Build a runnable MonetDB `mserver5` without regressing the native server.
2. Build a single-process TPC-H runner using the same MonetDB/GDK engine for
   gem5 syscall-emulation experiments.
3. Form device-legal regions automatically from CXL-resident MonetDB accesses.
4. Compile outlined regions into an RV64 device ELF and emit a deterministic
   kernel manifest and dispatch table.
5. Execute those kernels on a separate simulated RV64 timing core connected to
   shared CXL memory.
6. Preserve host authority: the device performs read-only address generation,
   memory traversal, and cache preparation while the host retains SQL
   computation and externally visible state changes.
7. Validate all TPC-H queries at SF=0.01 and prove device execution for Q1, Q6,
   Q9, Q11, and Q13.
8. Preserve the CIRA descriptor and completion semantics needed for a later
   IA-780I/Vortex implementation.

## Non-Goals

- Cycle-accurate modeling of Vortex SIMT microarchitecture.
- Running the final SF=1 performance campaign in the first implementation.
- Moving SQL parsing, transaction management, locks, allocation, regular
  expression evaluation, or other host-only services onto RV64.
- Treating forced SF=0.01 debug offloads as evidence that a region is
  profitable.
- Using mock profiles or inferred completion as benchmark evidence.
- Replacing the authoritative MonetDB server with a synthetic kernel-only
  benchmark.

## System Architecture

### Build Products

The flow produces five reviewable products:

1. `mserver5`, built for x86 and runnable outside gem5.
2. `monetdb-tpch-cira`, a single-process x86 TPC-H runner linked to the same
   MonetDB/GDK engine. It avoids requiring an independently scheduled mclient
   and simulated socket setup in the initial syscall-emulation flow.
3. An RV64 device ELF containing the firmware service loop and outlined
   kernels.
4. A generated kernel manifest mapping stable kernel IDs to source locations,
   query attribution, argument layouts, ELF symbols, and legality metadata.
5. An `ALL`-ISA gem5 binary and heterogeneous MonetDB configuration containing
   both x86 and RV64 CPUs.

### Simulated Topology

The simulation uses one gem5 `System` built with all ISA decoders, an x86 host
CPU, and an RV64 timing CPU. The x86 CPU owns the syscall-emulation workload
and runs the TPC-H runner. The RV64 CPU is not assigned a second SE `Process`:
a dedicated device loader places the RV64 ELF segments into device instruction
memory, initializes its PC, stack, argument registers, and thread context, and
then starts the bare-metal firmware. The firmware makes no host Linux syscalls.
This avoids relying on gem5 to attach two incompatible ISA-specific SE
workloads to one system.

The topology contains three distinct memory classes:

- host-local memory for executable code, stacks, runtime metadata, and other
  non-dataset state; and
- device-local instruction and scratch memory for RV64 firmware and kernel
  code; and
- a shared CXL arena for MonetDB BAT/column/hash data, the 8 KiB CIRA control
  window, job argument slots, and completion cache lines.

Host and device do not exchange virtual pointers. A job contains offsets within
the shared CXL arena. Each side translates an offset using its own mapped arena
base. Bounds and alignment are checked before the device dereferences an
offset.

The RV64 core's normal loads reach the device side of the CXL-memory model.
Compiler-inserted cache-install operations use the modeled CXL.cache/DCOH path
to place relevant lines in the host cache hierarchy. An ordinary RV64 load is
not counted as a host cache installation.

### Host/Device Protocol

The existing `runtime/include/cira_cxl_job.h` layout remains the protocol
authority. Generic calls use a stable kernel ID rather than a host function
pointer. If necessary, the `CALL` payload is versioned to add the kernel ID,
argument-block offset, argument length, completion offset, and flags without
silently changing the old layout.

Submission order is:

1. Write the argument payload.
2. Publish the argument-slot header.
3. Publish the doorbell fields.
4. Write the doorbell sequence last as the commit.

Device completion uses the reverse publication rule: result and status fields
are written first, then completion magic/sequence is written last. The RV64
firmware validates magic, ABI version, kernel ID, sequence, sizes, alignment,
and arena bounds before dispatch.

## Compiler Design

### Remote-Data Provenance

Region formation will use explicit CXL provenance. Values are remote when they
derive from a CIRA handle, a designated remote address space, an annotated
MonetDB allocation, or a frontend/profile annotation that resolves to the
shared CXL arena. Arbitrary non-stack heap access is not sufficient in the
general compiler mode.

The SF=0.01 runner will allocate the TPC-H dataset's BAT/column/hash storage
from the CXL arena. Control structures that must be device-visible are copied
or represented through explicit argument records rather than relying on a
host virtual address.

### Candidate Formation

For each remote memory read, the compiler walks backward from its address
operands and includes:

- pure integer and floating-point address computation;
- compares, selects, casts, and GEP/member/stride operations;
- loop induction and supported loop-carried address state; and
- dependent reads required to calculate the next address.

Constants are rematerialized. External pure values become live-ins. Calls,
allocation, I/O, locks, unsupported atomics, unmodeled side effects, and
externally visible writes terminate the slice.

The device slice is read-only in the first implementation. The original host
operations remain authoritative and continue to execute. This makes timeout or
device rejection recoverable without rolling back database state.

### Region Merging and Scheduling

Candidates within the same loop or control region are compared by operation
set, live-ins, dominance, and root accesses. Overlapping compatible slices are
merged so that a loop does not submit one job for every individual load.
Merging is rejected when it introduces a side effect, an unsupported control
dependency, or incompatible synchronization requirements.

Submission is placed at the earliest dominating point at which every live-in
is available. Synchronization is placed at the first host operation that
requires the prepared access or at an explicit phase boundary. The compiler
must demonstrate a non-empty host interval between submission and wait; an
immediate submit/wait pair is retained only as a correctness diagnostic and is
not counted as overlap.

### Outlining and Kernel Identity

Each accepted region produces:

- an x86 submission stub;
- a typed, versioned argument structure stored in shared CXL memory;
- an RV64 function with no host ABI dependencies;
- a stable kernel ID derived from module identity, source location, function,
  and region identity; and
- a manifest record used by the linker, firmware dispatch table, test harness,
  and result reporter.

The RV64 kernel contains only operations admitted by the device-legality
analysis. Device linkage fails if unresolved host symbols remain. The compiler
or device-lowering pass inserts explicit cache-line installation for root lines
that the host is expected to consume.

### Target Queries

The initial query attribution is:

- Q1: aggregation-side memory traversal;
- Q6: sequential predicate scan;
- Q9: hash-chain/join traversal;
- Q11: join and candidate traversal; and
- Q13: string/grouping data traversal, with regex/string-library calls kept on
  the host unless separately proven device-legal.

The mechanism is not hard-coded to these query numbers. Query attribution is
metadata for validation; region formation remains source- and dependence-based.

### Selection Modes

The compiler exposes two clearly labeled modes:

- forced-debug mode, which offloads every legal selected candidate at SF=0.01
  to exercise the complete path; and
- measured mode, which applies profitability decisions from real gem5
  measurements.

Forced-debug results establish correctness and execution coverage only. They
must not be described as profitable or performance-improving.

## MonetDB Build and Runner

The build keeps native fallbacks for unrelated MonetDB translation units that
ClangIR cannot yet lower. The targeted GDK translation units must pass through
ClangIR, region formation, host lowering, and RV64 outlining. A targeted unit
may fall back only when the build fails closed and reports that the required
query no longer has device coverage; silent fallback is forbidden.

The normal `mserver5` build proves that the full server still compiles and can
start. The gem5 runner uses MonetDB's embedded/in-process execution path with
the same GDK engine, loads SF=0.01 data, runs Q1-Q22, and emits canonical query
results and per-query ROI markers. It does not replace the full-server build
gate.

## Runtime and Firmware

The x86 runtime owns shared-arena allocation, descriptor construction, sequence
allocation, submission, completion waits, timeout policy, and fallback status.
It records the kernel ID and query/region metadata with every job.

The RV64 firmware:

1. Initializes the arena and dispatch-table base.
2. Polls or waits for a doorbell sequence change.
3. Reads a coherent descriptor snapshot.
4. Validates the descriptor and resolves the kernel ID.
5. Calls the RV64 kernel with an arena-relative argument block.
6. Publishes completion and returns to the service loop.

Firmware and kernels are built without host libc or MonetDB runtime
dependencies. Minimal memory, fence, timing, cache-install, and completion
helpers are provided by the device runtime.

## Failure Handling

The following conditions are explicit failures:

- ABI magic or version mismatch;
- stale or torn sequence state;
- unknown kernel ID;
- argument-layout or payload-size mismatch;
- misaligned or out-of-range arena offset;
- unresolved device symbol;
- RV64 exception or illegal instruction;
- completion timeout;
- submitted/completed count mismatch;
- query-result mismatch; or
- a required target query that silently executes only the host fallback.

Because initial device regions are read-only, the host can continue its
authoritative path after a rejected or timed-out job. The run remains correct
but is marked as an offload failure and cannot satisfy the acceptance gate.

## Verification Strategy

### Compiler Tests

Focused MLIR/CIR tests cover:

- remote provenance and non-remote negative cases;
- dependent-load slicing;
- loop-carried address state;
- constants and live-ins;
- rejection of calls, writes, locks, and unsupported atomics;
- compatible region merging and incompatible merge rejection;
- deterministic kernel IDs and manifest output;
- submission hoisting and wait placement; and
- x86 submission plus RV64 kernel generation from one region.

### ABI and Firmware Tests

Host and RV64 builds assert descriptor sizes, offsets, alignment, constants,
and sequence ordering. Synthetic gem5 tests submit valid and malformed jobs and
cover successful dispatch, unknown IDs, bad bounds, bad versions, stale
sequences, device exceptions, and timeouts.

A firmware smoke test must retire RV64 instructions inside a known kernel
symbol range, generate CXL requests, write completion, and return to the
service loop.

### MonetDB Correctness Tests

The full native server must build and pass a startup/version smoke test. The
TPC-H runner executes Q1-Q22 in baseline and forced-debug offload modes at
SF=0.01. Result rows are canonicalized where SQL ordering is unspecified, then
row counts, nulls, values, and deterministic digests are compared. Numeric
comparison follows MonetDB's SQL type semantics; the validator does not hide a
result mismatch behind a broad floating-point tolerance.

Q1, Q6, Q9, Q11, and Q13 additionally require:

- at least one submitted and completed kernel;
- zero failed and pending kernels;
- matching submitted and completed sequences;
- retired RV64 instructions within the expected kernel symbols;
- nonzero CXL reads; and
- nonzero coherent host cache-line installations.

### Matched gem5 Runs

Baseline and offload runs use the same x86 CPU model, core count, clock,
caches, memory sizes, dataset, query inputs, and CXL link parameters. Generated
`config.ini` is checked directly for the selected topology and link delay.

The report includes simulated ROI ticks, host instructions, RV64 instructions,
CXL traffic, cache installations, job latency, overlap cycles, failures, and
artifact paths. SF=0.01 has no required speedup threshold. Performance claims
begin only after correctness and execution coverage pass and are followed by a
separately approved SF=1 campaign.

## Acceptance Criteria

The first milestone is complete only when all of the following hold:

1. Both repositories preserve unrelated dirty worktree changes.
2. CXLMemUring compiler and runtime tests pass.
3. `build/ALL/gem5.opt` builds with the heterogeneous extensions.
4. `mserver5` builds and passes its smoke check.
5. The x86 TPC-H runner and RV64 firmware/device ELF build reproducibly.
6. Q1-Q22 match baseline results at SF=0.01.
7. Q1, Q6, Q9, Q11, and Q13 satisfy every device-execution evidence gate.
8. Descriptor, completion, and gem5 statistics show no lost or stranded jobs.
9. The final report labels the device as an RV64 Vortex surrogate and separates
   correctness/debug evidence from performance evidence.

Passing only compilation, submission, host fallback, or simulator process exit
does not satisfy the milestone.
