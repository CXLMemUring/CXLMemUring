//
// Created by yangyw on 8/4/24.
//

#ifndef CIRA_PASSES_H
#define CIRA_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "Dialect/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createRemoteMemDPSPass();
std::unique_ptr<Pass> createRemoteMemSearchRemotePass();

#define GEN_PASS_REGISTRATION
#include "Dialect/Transforms/Passes.h.inc"
} // namespace mlir

#endif // CIRA_PASSES_H
