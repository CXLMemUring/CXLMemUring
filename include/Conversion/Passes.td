#define REMOTE_MEM_PASSES

include "mlir/Pass/PassBase.td"

def CIRA : Pass<"convert-target-to-remote", "ModuleOp"> {
  let summary = "Will calculate the complexity and mark remote";
  let description = "converts target to cira dialect";
  let constructor = "mlir::cira::createCIRAPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect", 
    "cira::RemoteMemDialect", 
    "func::FuncDialect", 
    "memref::MemRefDialect", 
    "scf::SCFDialect"
  ];
}

// Expose the ClangIR-to-CIRA pipeline entry (front-end like tagging + offload)
def CIRToCira : Pass<"cir-to-cira", "ModuleOp"> {
  let summary = "Convert/annotate ClangIR/SCF patterns into CIRA offload candidates";
  let description = [{
    This pass identifies common C/C++ graph-processing patterns (array scans,
    nested loops, indirect loads) in ClangIR/SCF and either annotates them as
    candidates for remote execution or wraps them in CIRA offload constructs.
    Initially non-destructive and attribute-based for safe frontend testing.
  }];
  let constructor = "mlir::cira::createCIRToCiraPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "cira::RemoteMemDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def ConvertCiraToLLVM : Pass<"convert-cira-to-llvm", "ModuleOp"> {
  let summary = "Convert Cira dialect operations to LLVM dialect";
  let description = [{
    This pass converts operations from the Cira dialect to LLVM dialect,
    handling offload operations, loads, stores, and other graph processing
    primitives. Auto-detects target architecture from module attributes.
  }];
  let constructor = "mlir::cira::createConvertCiraToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def ConvertCiraToLLVMX86 : Pass<"convert-cira-to-llvm-x86", "ModuleOp"> {
  let summary = "Convert Cira dialect to LLVM for x86 target";
  let description = [{
    This pass converts Cira dialect operations to LLVM dialect specifically
    for x86-64 architectures, optimizing for computational workloads.
  }];
  let constructor = "mlir::cira::createConvertCiraToLLVMX86Pass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def ConvertCiraToLLVMARM : Pass<"convert-cira-to-llvm-arm", "ModuleOp"> {
  let summary = "Convert Cira dialect to LLVM for ARM target";
  let description = [{
    This pass converts Cira dialect operations to LLVM dialect specifically
    for ARM/AArch64 architectures, optimizing for memory management tasks.
  }];
  let constructor = "mlir::cira::createConvertCiraToLLVMARMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def ConvertCiraToLLVMHetero : Pass<"convert-cira-to-llvm-hetero", "ModuleOp"> {
  let summary = "Convert Cira dialect to LLVM for heterogeneous execution";
  let description = [{
    This pass converts Cira dialect operations to LLVM dialect for
    heterogeneous execution across x86 (computation) and ARM (memory management)
    processors, partitioning functions based on their workload characteristics.
  }];
  let constructor = "mlir::cira::createConvertCiraToLLVMHeteroPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}
