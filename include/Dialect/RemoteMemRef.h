#ifndef CIRA_REMOTEMEMREF_H
#define CIRA_REMOTEMEMREF_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
namespace cira {

/// Storage class for RemoteMemRefType
namespace detail {
/// Storage class for RemoteMemRefType
struct RemoteMemRefTypeStorage : public TypeStorage {
  RemoteMemRefTypeStorage(Type elementType, unsigned cacheID)
      : elementType(elementType), cacheID(cacheID) {}

  /// The hash key used for uniquing.
  using KeyTy = std::pair<Type, unsigned>;
  
  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key.first == elementType && key.second == cacheID;
  }
  
  /// Define a hash function for the key type.
  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }
  
  /// Define a construction method for creating a new instance of this storage.
  static RemoteMemRefTypeStorage *construct(TypeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<RemoteMemRefTypeStorage>())
        RemoteMemRefTypeStorage(key.first, key.second);
  }

  Type elementType;
  unsigned cacheID;
};
} // namespace detail

class RemoteMemRefType : public mlir::Type::TypeBase<RemoteMemRefType, mlir::Type,
                                               detail::RemoteMemRefTypeStorage> {
public:
  using Base::Base;

  // Type name for MLIR type system
  static constexpr StringLiteral name = "cira.remote_memref";

  static RemoteMemRefType get(Type elementType, unsigned cacheID = 0);
  
  static bool classof(Type type);
  
  static bool isValidElementType(Type elementType);
  
  Type getElementType() const;
  
  unsigned getCacheID() const;

  static TypeID getTypeID() {
    static TypeID typeID = TypeID::get<RemoteMemRefType>();
    return typeID;
  }
  
  // 添加 getCanRemote 方法
  bool getCanRemote() const {
    return true;
  }
};

} // namespace cira
} // namespace mlir

#endif // CIRA_REMOTEMEMREF_H 