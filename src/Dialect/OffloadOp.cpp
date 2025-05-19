#include "Dialect/OffloadOp.h"
#include "Dialect/RemoteMem.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::cira;

// OffloadOp方法实现
namespace mlir {
namespace cira {
namespace offload_impl {

// Parse方法实现
ParseResult parseOffloadOp(OpAsmParser &parser, OperationState &result) {
    Type resultType;
    if (parser.parseType(resultType))
        return failure();
    
    SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
    if (parser.parseOperandList(operands))
        return failure();
    
    SmallVector<Type, 4> operandTypes;
    if (parser.parseColonTypeList(operandTypes))
        return failure();
    
    if (parser.resolveOperands(operands, operandTypes, parser.getNameLoc(), result.operands))
        return failure();
    
    result.addTypes({resultType});
    
    return success();
}

// Print方法实现
void printOffloadOp(OpAsmPrinter &p, Operation *op) {
    p << " ";
    p.printOperands(op->getOperands());
    p << " : ";
    llvm::interleaveComma(op->getOperandTypes(), p);
}

// 验证方法实现
LogicalResult verifyOffloadOp(Operation *op) {
    return success();
}

} // namespace offload_impl
} // namespace cira
} // namespace mlir 