#ifndef CIRA_OPS_H
#define CIRA_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "Dialect/RemoteMemDialect.h"
#include "Dialect/RemoteMemRef.h"

// Include the generated types (Handle, Future, Stream)
#define GET_TYPEDEF_CLASSES
#include "Dialect/RemoteMemTypes.h.inc"

// Forward declarations
namespace mlir {
namespace cira {
class RemoteMemDialect;
} // namespace cira
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/CiraOps.h.inc"

#endif // CIRA_OPS_H