#ifndef CIRA_TO_LLVM_H
#define CIRA_TO_LLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace cira {

/// Populate the given list with patterns that convert from Cira to LLVM.
void populateCiraToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);

/// Create a pass to convert Cira operations to the LLVM dialect.
std::unique_ptr<Pass> createConvertCiraToLLVMPass();

/// Create a pass to convert Cira operations to LLVM for x86 target.
std::unique_ptr<Pass> createConvertCiraToLLVMX86Pass();

/// Create a pass to convert Cira operations to LLVM for ARM target.
std::unique_ptr<Pass> createConvertCiraToLLVMARMPass();

/// Create a pass to convert Cira operations to LLVM for heterogeneous execution.
std::unique_ptr<Pass> createConvertCiraToLLVMHeteroPass();

} // namespace cira
} // namespace mlir

#endif // CIRA_TO_LLVM_H