#ifndef CIRA_OFFLOADOP_H
#define CIRA_OFFLOADOP_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
namespace cira {

// OffloadOp前置声明，实际定义在RemoteMem.h.inc中
class OffloadOp;

// 添加Op方法声明，实际实现在OffloadOp.cpp中  
namespace offload_impl {
ParseResult parseOffloadOp(OpAsmParser &parser, OperationState &result);
void printOffloadOp(OpAsmPrinter &p, Operation *op);
LogicalResult verifyOffloadOp(Operation *op);
}

} // namespace cira
} // namespace mlir

#endif // CIRA_OFFLOADOP_H 