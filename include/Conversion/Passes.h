#ifndef MLIR_REMOTE_MEM_LOWERING_PASSES_H
#define MLIR_REMOTE_MEM_LOWERING_PASSES_H

#include "mlir/Pass/Pass.h"
#include "Conversion/CIRA.h"
#include "Conversion/CiraToLLVM.h"
#include "Dialect/RemoteMem.h"

namespace mlir {
class RewritePatternSet;
namespace cira {

}
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
}

#endif