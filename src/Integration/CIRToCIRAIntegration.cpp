//===- CIRToCIRAIntegration.cpp - Integration of CIRA into CIR pipeline ===//
//
// This file implements integration of CIRA transformation into the CIR to
// MLIR pipeline for automatic offloading analysis and transformation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Module.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "Conversion/CIRA.h"
#include "Conversion/CiraToLLVM.h"
#include "Dialect/RemoteMem.h"
#include "Dialect/FunctionUtils.h"

namespace mlir {
namespace cira {

/// Pipeline function to lower CIR to CIRA to LLVM
/// This integrates with ClangIR's lowering pipeline
void buildCIRToCIRAToLLVMPipeline(mlir::PassManager &pm, bool enableOffloading) {
  // Step 1: Lower CIR to standard MLIR dialects (SCF, MemRef, etc.)
  // This is handled by ClangIR's existing passes

  // Step 2: Apply CIRA transformation for remote memory offloading
  if (enableOffloading) {
    pm.addPass(cira::createCIRAPass());
  }

  // Step 3: Lower CIRA-transformed code to LLVM
  // Can choose between x86, ARM, or heterogeneous targets
  pm.addPass(cira::createConvertCiraToLLVMHeteroPass());
}

/// Hook function to be called from ClangIR's pipeline
/// This should be invoked after CIR→MLIR but before MLIR→LLVM
void applyCIRATransformation(mlir::ModuleOp module, bool enableAnalysis) {
  mlir::PassManager pm(module.getContext());

  // Run CIRA analysis and transformation
  pm.addPass(cira::createCIRAPass());

  if (mlir::failed(pm.run(module))) {
    module.emitError("CIRA transformation failed");
  }
}

/// Analysis function to determine if a module would benefit from CIRA
bool shouldApplyCIRATransformation(mlir::ModuleOp module) {
  bool hasGraphPatterns = false;
  bool hasLargeLoops = false;

  module.walk([&](mlir::Operation *op) {
    // Check for graph-like access patterns
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Check trip count
      if (auto constantUpperBound = forOp.getUpperBound().getDefiningOp<arith::ConstantOp>()) {
        if (auto attr = dyn_cast<IntegerAttr>(constantUpperBound.getValue())) {
          if (attr.getInt() > 1000) {
            hasLargeLoops = true;
          }
        }
      }

      // Check for indirect memory accesses (graph patterns)
      forOp.walk([&](memref::LoadOp loadOp) {
        // Check if the load index comes from another load (indirect access)
        for (auto index : loadOp.getIndices()) {
          if (index.getDefiningOp<memref::LoadOp>()) {
            hasGraphPatterns = true;
            return;
          }
        }
      });
    }
  });

  return hasGraphPatterns || hasLargeLoops;
}

} // namespace cira
} // namespace mlir