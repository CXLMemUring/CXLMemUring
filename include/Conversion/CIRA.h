#ifndef REMOTE_TARGET_TO_REMOTE_MEM_H
#define REMOTE_TARGET_TO_REMOTE_MEM_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringSet.h"
#include <memory>

namespace mlir {
class Pass;
class Value;
class Location;
class RewritePatternSet;
class ConversionPatternRewriter;

#define GEN_PASS_DECL_CIRA
#include "Conversion/Passes.h.inc"

namespace cira {
class RemoteMemDialect;

void populateCIRAPatterns(MLIRContext *ctx, RewritePatternSet &patterns);
void populateSCFCIRAPatterns(MLIRContext *ctx, RewritePatternSet &patterns);
void populateSCFToCiraPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

std::unique_ptr<Pass> createCIRAPass();
} // namespace cira
} // namespace mlir

#endif