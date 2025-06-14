//
// Created by yangyw on 8/5/24.
//
#include "mlir/IR/BuiltinTypes.h"
#include "Dialect/RemoteMem.h"
#include "Dialect/RemoteMemDialect.h"
#include "Dialect/RemoteMemRef.h"
#include "Dialect/OffloadOp.h"
#include "Dialect/FunctionUtils.h"
#include "Dialect/CiraOps.h"

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

// Include the generated dialect definitions
#include "Dialect/RemoteMemDialect.cpp.inc"