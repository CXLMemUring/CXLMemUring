//
// Created by yangyw on 8/5/24.
//
#include "compat/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "Dialect/RemoteMemDialect.h"
#include "Dialect/RemoteMemRef.h"
#include "Dialect/OffloadOp.h"
#include "Dialect/FunctionUtils.h"
#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"

using namespace mlir;
using namespace mlir::cira;

// 注册类型和操作
void RemoteMemDialect::initialize() {
    registerTypes();
    
    // Register Cira operations
    addOperations<
#define GET_OP_LIST
#include "Dialect/CiraOps.cpp.inc"
    >();
}

void RemoteMemDialect::registerTypes() {
    addTypes<RemoteMemRefType>();
}

Attribute RemoteMemDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
    // Basic attribute parsing - for now just return failure
    parser.emitError(parser.getNameLoc(), "custom attributes not supported yet");
    return {};
}

void RemoteMemDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
    // Basic attribute printing - for now just emit error
    printer << "unknown_attr";
}

Type RemoteMemDialect::parseType(DialectAsmParser &parser) const {
    // Basic type parsing - for now just return failure
    parser.emitError(parser.getNameLoc(), "custom types not supported yet");
    return {};
}

void RemoteMemDialect::printType(Type type, DialectAsmPrinter &printer) const {
    // Basic type printing - for now just emit error
    printer << "unknown_type";
}

// Include the generated dialect definitions
#include "Dialect/RemoteMemDialect.cpp.inc"