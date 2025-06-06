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

def ConvertCiraToLLVM : Pass<"convert-cira-to-llvm", "ModuleOp"> {
  let summary = "Convert Cira dialect operations to LLVM dialect";
  let description = [{
    This pass converts operations from the Cira dialect to LLVM dialect,
    handling offload operations, loads, stores, and other graph processing
    primitives.
  }];
  let constructor = "mlir::cira::createConvertCiraToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}