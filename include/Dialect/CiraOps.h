#ifndef CIRA_OPS_H
#define CIRA_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/RemoteMem.h"
#include "Dialect/RemoteMemRef.h"

namespace mlir {
namespace cira {

//===----------------------------------------------------------------------===//
// Cira Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/CiraOps.h.inc"

} // namespace cira
} // namespace mlir

#endif // CIRA_OPS_H