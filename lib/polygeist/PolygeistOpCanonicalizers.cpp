#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace polygeist {

// AlternativesOp canonicalization patterns
void AlternativesOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  // Empty implementation
}

// GetDeviceGlobalOp memory effects
void GetDeviceGlobalOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Empty implementation - implicitly has no effects
}

// GetDeviceGlobalOp symbol verification
LogicalResult GetDeviceGlobalOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Empty verification - always succeeds
  return success();
}

// GetFuncOp canonicalization patterns
void GetFuncOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  // Empty implementation
}

// NoopOp memory effects
void NoopOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Empty implementation - implicitly has no effects
}

// UndefOp canonicalization patterns
void UndefOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  // Empty implementation
}

} // namespace polygeist
} // namespace mlir 