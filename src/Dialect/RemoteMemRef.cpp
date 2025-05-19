#include "Dialect/RemoteMemRef.h"
#include "Dialect/RemoteMemDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::cira;

// 实现 get 方法
RemoteMemRefType RemoteMemRefType::get(Type elementType, unsigned cacheID) {
    return Base::get(elementType.getContext(), elementType, cacheID);
}

// 实现 classof 方法
bool RemoteMemRefType::classof(Type type) {
    return type.isa<RemoteMemRefType>();
}

// 实现 isValidElementType 方法
bool RemoteMemRefType::isValidElementType(Type elementType) {
    if (!elementType) return false;
    if (!elementType.isa<mlir::MemRefType>() && 
        !elementType.isa<LLVM::LLVMPointerType>() && 
        !elementType.isa<mlir::UnrankedMemRefType>()) 
        return false;
    return true;
}

// 实现 getElementType 方法
Type RemoteMemRefType::getElementType() const {
    auto *storage = static_cast<detail::RemoteMemRefTypeStorage *>(getImpl());
    return storage->elementType;
}

// 实现 getCacheID 方法
unsigned RemoteMemRefType::getCacheID() const {
    auto *storage = static_cast<detail::RemoteMemRefTypeStorage *>(getImpl());
    return storage->cacheID;
} 