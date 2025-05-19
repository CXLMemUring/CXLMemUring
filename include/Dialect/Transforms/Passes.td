#ifndef TRANSFORM_PASSES
#define TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def RMEMSearchRemote : Pass<"rmem-search-remote", "::mlir::ModuleOp"> {
  let summary = "Search for loops that can be offloaded to remote processing";
  let description = [{
    This pass searches for loops that can be offloaded to remote execution.
    It extracts the loop body into a separate function and replaces the loop
    with a call to that function.
  }];
}

#endif
