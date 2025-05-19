//
// Created by yangyw on 8/5/24.
//
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "Dialect/RemoteMem.h"
#include "Dialect/RemoteMemDialect.h"
#include "Dialect/RemoteMemRef.h"
#include "Dialect/OffloadOp.h"
#include "Dialect/FunctionUtils.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace mlir;
using namespace mlir::cira;

// 注册类型和操作
void RemoteMemDialect::initialize() {
    registerTypes();
    
    // Tablegen会自动注册操作，不需要手动添加
    // addOperations<OffloadOp>();
}

void RemoteMemDialect::registerTypes() {
    addTypes<RemoteMemRefType>();
}

::mlir::Attribute RemoteMemDialect::parseAttribute(mlir::DialectAsmParser &parser, mlir::Type type) const {
    return nullptr;
}

void RemoteMemDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const {
    // Empty implementation
}

// 添加到全局命名空间，便于其他文件引用
namespace mlir {
namespace cira {
class OffloadOp;
} // namespace cira
} // namespace mlir