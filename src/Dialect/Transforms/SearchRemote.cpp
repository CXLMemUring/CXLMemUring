//
// Created by yangyw on 8/5/24.
//
#include "Dialect/RemoteMem.h"
#include "Dialect/Transforms/Passes.h"
#include "Dialect/WorkloadAnalysis.h"
#include "Dialect/FunctionUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/TypeSize.h"

#include <set>

namespace mlir {
// 首先定义Pass类，确保在Tablegen生成代码之前
namespace {
class RMEMSearchRemotePass;
} // end anonymous namespace

// 然后包含tablegen生成的头文件
#define GEN_PASS_DEF_RMEMSEARCHREMOTE
#include "Dialect/Transforms/Passes.h.inc"

using namespace mlir;

// 避免使用匿名命名空间，因为我们需要外部定义createRMEMSearchRemote函数
namespace rmem_search_remote_impl {

// Pass class implementing the loop extraction and remote execution transformation
class RMEMSearchRemotePass
    : public impl::RMEMSearchRemoteBase<RMEMSearchRemotePass> {
public:
  RMEMSearchRemotePass() = default;

  // Recursive function to find paths in the data flow graph that depend on the loop induction variable
  std::set<Operation *> addrPathDFS(Operation *op, scf::ForOp loop) {
    std::set<Operation *> search;
    for (OpOperand &opd : op->getOpOperands()) {
      if (opd.get() == loop.getInductionVar())
        search.insert(op);
      else {
        Operation *def = opd.get().getDefiningOp();
        if (def && def->getBlock() == loop.getBody()) {
          auto dfs = addrPathDFS(def, loop);
          if (dfs.size()) {
            dfs.insert(op);
            search.merge(dfs);
          }
        }
      }
    }
    return search;
  }

  // Find values used inside the loop that are defined outside the loop
  llvm::SetVector<mlir::Value> analyzeValueUses(mlir::scf::ForOp forOp) {
    llvm::SetVector<mlir::Value> capturedValues;
    mlir::Region &loopBody = forOp.getRegion();

    forOp.walk([&](mlir::Operation *op) {
      for (mlir::Value operand : op->getOperands()) {
        mlir::Operation *definingOp = operand.getDefiningOp();

        // Check if the operand is defined outside the loop
        // and is not a result of an operation within the loop
        if (!loopBody.isAncestor(operand.getParentRegion()) &&
            (definingOp == nullptr ||
             !loopBody.isAncestor(definingOp->getParentRegion()))) {
          capturedValues.insert(operand);
        }
      }
    });

    // Remove loop induction variable and loop-defined values from captured values
    capturedValues.remove(forOp.getInductionVar());
    for (mlir::Value arg : forOp.getRegionIterArgs()) {
      capturedValues.remove(arg);
    }

    return capturedValues;
  }

  // Extract the loop body into a separate function
  mlir::func::FuncOp
  extractLoopBody(mlir::scf::ForOp forOp,
                  const llvm::SetVector<mlir::Value> &capturedValues,
                  mlir::OpBuilder &builder) {
    mlir::Block *body = forOp.getBody();

    // Prepare function type
    llvm::SmallVector<mlir::Type, 4> argTypes;
    argTypes.push_back(builder.getIndexType()); // Induction variable
    for (mlir::Value val : capturedValues) {
      argTypes.push_back(val.getType());
    }
    auto funcType = builder.getFunctionType(argTypes, {});

    // Create a new function for the remote execution
    auto funcName = mlir::cira::getNextRemoteAccessName();
    auto funcOp =
        mlir::func::FuncOp::create(forOp.getLoc(), funcName, funcType);

    // Add entry block to the function
    auto *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Create a mapping from old values to new function arguments
    mlir::BlockAndValueMapping mapping;
    mapping.map(forOp.getInductionVar(), entryBlock->getArgument(0));
    for (size_t i = 0; i < capturedValues.size(); ++i) {
      mapping.map(capturedValues[i], entryBlock->getArgument(i + 1));
    }

    // Clone loop body operations, remapping the values
    for (auto &op : body->getOperations()) {
      if (!mlir::isa<mlir::scf::YieldOp>(op)) {
        builder.clone(op, mapping);
      }
    }

    builder.create<mlir::func::ReturnOp>(forOp.getLoc());
    return funcOp;
  }

  // Replace the loop with a call to the remote function
  void replaceWithRemoteCall(mlir::scf::ForOp forOp,
                             mlir::func::FuncOp remoteFunc,
                             const llvm::SetVector<mlir::Value> &capturedValues,
                             mlir::OpBuilder &builder) {
    builder.setInsertionPoint(forOp);

    auto newForOp = builder.create<mlir::scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep());

    builder.setInsertionPointToStart(newForOp.getBody());

    llvm::SmallVector<mlir::Value, 4> callOperands;
    callOperands.push_back(newForOp.getInductionVar());
    callOperands.append(capturedValues.begin(), capturedValues.end());

    builder.create<mlir::func::CallOp>(forOp.getLoc(), remoteFunc,
                                        callOperands);

    forOp.erase();
  }

  // Main entry point for the pass
  void runOnOperation() override {
    ModuleOp mop = cast<ModuleOp>(getOperation());

    mlir::OpBuilder builder(mop.getContext());

    llvm::SmallVector<mlir::scf::ForOp, 4> forOpsToReplace;
    mop.walk([&](mlir::scf::ForOp forOp) {
      //            if (isComplexEnough(forOp)) {
      forOpsToReplace.push_back(forOp);
      //            }
    });

    for (auto forOp : forOpsToReplace) {
      llvm::SetVector<mlir::Value> capturedValues = analyzeValueUses(forOp);
      mlir::func::FuncOp remoteFunc =
          extractLoopBody(forOp, capturedValues, builder);

      // Add the new function to the module
      mop.push_back(remoteFunc);

      replaceWithRemoteCall(forOp, remoteFunc, capturedValues, builder);
    }
  }
};

} // namespace rmem_search_remote_impl

// 向后兼容函数
std::unique_ptr<Pass> createRemoteMemSearchRemotePass() {
  return std::make_unique<rmem_search_remote_impl::RMEMSearchRemotePass>();
}

} // namespace mlir
