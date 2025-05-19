//
// Created by yangyw on 8/4/24.
//

#ifndef CIRA_REMOTEMEMTOLLVM_H
#define CIRA_REMOTEMEMTOLLVM_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
void populateRemoteMemToLLVMPatterns(mlir::RewritePatternSet &patterns);
std::unique_ptr<Pass> createRemoteMemToLLVMPass();
} // namespace mlir

#endif // CIRA_REMOTEMEMTOLLVM_H
