//
// Created by yangyw on 8/5/24.
//
#include "Dialect/WorkloadAnalysis.h"
#include "Dialect/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "Dialect/RemoteMem.h"
#include "Dialect/FunctionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

/*
return value
  if upper or lower bound unknown: uncertain
  else if step unknown: upper - lower
  else ceil (upper - lower) / step
*/

unsigned WorkloadComplexityAnalyzer::visitSCFForOp(scf::ForOp forOp) {
    auto upperBoundCst =
        forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto lowerBoundCst =
        forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto stepCst = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
    // uncertain
    if (!upperBoundCst || !lowerBoundCst)
        return WorkloadComplexityAnalyzer::uncertain;

    int64_t ub = upperBoundCst.value();
    int64_t lb = lowerBoundCst.value();
    int64_t step = stepCst ? stepCst.value() : 1;
    int64_t numIteration = llvm::divideCeil(ub - lb, step);

    unsigned complexity = visitBlock(forOp.getBody()) * numIteration;
    complexity = complexity < WorkloadComplexityAnalyzer::uncertain ? complexity : WorkloadComplexityAnalyzer::uncertain;
    rel[forOp] = complexity;
    return complexity;
}

unsigned WorkloadComplexityAnalyzer::visitMLIRFuncOp(func::FuncOp op) {
    unsigned complexity = 0;
    for (auto &block : op.getFunctionBody().getBlocks()) {
        complexity += visitBlock(&block);
        if (complexity > WorkloadComplexityAnalyzer::uncertain) {
            complexity = WorkloadComplexityAnalyzer::uncertain;
            break;
        }
    }
    rel[op] = complexity;
    return complexity;
}

unsigned WorkloadComplexityAnalyzer::visitOperation(Operation *op) {
    if (rel.find(op) != rel.end())
        return rel[op];
    if (auto For = dyn_cast<scf::ForOp>(op)) {
        return visitSCFForOp(For);
    }
    if (auto WHILE = dyn_cast<scf::WhileOp>(op)) {
        return WorkloadComplexityAnalyzer::uncertain;
    }
    if (auto Func = dyn_cast<func::FuncOp>(op)) {
        return visitMLIRFuncOp(Func);
    }
    if (auto IF = dyn_cast<scf::IfOp>(op)) {
        return std::min(visitBlock(IF.thenBlock()), visitBlock(IF.elseBlock()));
    }
    return 1;
}

unsigned WorkloadComplexityAnalyzer::visitBlock(Block *block) {
    unsigned complexity = 0;
    if (block->empty()) return 0;
    for (auto &op : block->getOperations()) {
        complexity += visitOperation(&op);
        if (complexity > WorkloadComplexityAnalyzer::uncertain) {
            complexity = WorkloadComplexityAnalyzer::uncertain;
            break;
        }
    }
    return complexity;
}

