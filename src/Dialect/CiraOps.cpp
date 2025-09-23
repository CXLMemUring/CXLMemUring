#include "compat/LLVM.h"
#include "Dialect/CiraOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::cira;

//===----------------------------------------------------------------------===//
// CiraCallOp
//===----------------------------------------------------------------------===//

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Basic symbol verification - check if the callee symbol exists
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/CiraOps.cpp.inc"