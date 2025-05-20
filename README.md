# CXLMemUring: a CXL memory offloaded runtime.
Can we replace the while loop usleep with coroutine and gain speed up?

```bash
$ time taskset -c 0 numactl -p 1 ./a.out
c=193843754

________________________________________________________
Executed in   4.00 secs    fish           external
   usr time    3.71 secs    1.09 millis    3.71 secs
   sys time    1.10 secs    0.14 millis    1.10 secs

$ time taskset -c 0 numactl -p 1 ./a.out
handle: 0x775e8c000b90
c=193843754

________________________________________________________
Executed in  141.85 secs    fish           external
   usr time    8.02 secs  706.00 micros    8.02 secs
   sys time   11.42 secs  567.00 micros   11.42 secs
```

**Notes**

- The repository implements a runtime and compiler infrastructure for experimenting with offloading computations to CXL-based remote memory.
- It defines a custom MLIR dialect (`RemoteMemDialect`) along with conversion and lowering passes to LLVM IR, plus a runtime using C++20 coroutines to asynchronously communicate with remote functions.
- Build configuration is via CMake and relies on MLIR/LLVM.

**Summary**

The project is organized around compiler passes, runtime code, and benchmarks.

1. **Compiler Infrastructure**
   - `include/` and `src/` define the **RemoteMemDialect** and related passes.
     - `RemoteMemRefType` stores metadata for remote memory references.
     - `driver.cpp` registers the dialect and passes with MLIR’s driver.
   - `src/Conversion` and `src/Lowering` supply pattern rewrites for converting high-level operations to remote memory operations and eventually to LLVM IR.
   - `lib/polygeist` provides supporting polygeist dialect code and passes.
2. **Runtime**
   - `runtime/` contains C++ implementations of lock‑free queues, channels, and coroutine-based tasks.
     The `Channel` template in `utils.h` implements a lightweight ring buffer for async communication.
   - Example runtime code (`affinity.cpp`) shows how remote calls are issued through coroutines and channels.
3. **Benchmarks and Tests**
   - `bench/` holds benchmark directories (e.g., `mcf/`) with a Makefile that can compile via MLIR tools.
   - `test/` contains simple C/C++ files and remote implementations for experimentation.
4. **Tools**
   - `tools/` provides placeholder command-line tools (`cgeist`, `polygeist-opt`) for interacting with MLIR passes.

**Pointers for New Contributors**

- Become familiar with **MLIR dialects and passes**. Study `include/Dialect/` headers and `src/Dialect` implementations to understand how remote memory operations are represented and lowered.
- Explore the **runtime** in `runtime/` to see how coroutines and channels manage communication with remote libraries.
- Review `src/driver.cpp` and `tools/polygeist-opt` to learn how passes are registered and executed.
- Building requires LLVM/MLIR; inspect the root `CMakeLists.txt` for configuration steps.
- To extend or modify benchmarks, check `bench/mcf` and the corresponding runtime code in `runtime/`.

Further learning could focus on writing new MLIR passes, experimenting with asynchronous offloading in the runtime, or integrating additional benchmarks.
