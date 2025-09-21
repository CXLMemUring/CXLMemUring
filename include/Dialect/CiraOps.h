#ifndef CIRA_OPS_H
#define CIRA_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/RemoteMemDialect.h"
#include "Dialect/RemoteMemRef.h"

// Forward declarations
namespace mlir {
namespace cira {
class RemoteMemDialect;
} // namespace cira
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/CiraOps.h.inc"

#endif // CIRA_OPS_H