//
// Created by yangyw on 8/5/24.
//

#include "Dialect/RemoteMemDialect.h"
#include "Dialect/CiraOps.h"
#include "Dialect/FunctionUtils.h"
#include "Dialect/OffloadOp.h"
#include "Dialect/RemoteMem.h"
#include "Dialect/RemoteMemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cira;

//===----------------------------------------------------------------------===//
// CIRA Type Definitions - Handle, Future, Stream
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cira {

/// HandleType - A pointer guaranteed to reside in CXL space
struct HandleTypeStorage : public TypeStorage {
  using KeyTy = Type;

  HandleTypeStorage(Type elementType) : elementType(elementType) {}

  bool operator==(const KeyTy &key) const { return elementType == key; }

  static HandleTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<HandleTypeStorage>()) HandleTypeStorage(key);
  }

  Type elementType;
};

class HandleType : public Type::TypeBase<HandleType, Type, HandleTypeStorage> {
public:
  using Base::Base;

  static HandleType get(Type elementType) {
    return Base::get(elementType.getContext(), elementType);
  }

  Type getElementType() const { return getImpl()->elementType; }

  static constexpr StringLiteral name = "cira.handle";
};

/// FutureType - A token representing an in-flight asynchronous load
struct FutureTypeStorage : public TypeStorage {
  using KeyTy = Type;

  FutureTypeStorage(Type valueType) : valueType(valueType) {}

  bool operator==(const KeyTy &key) const { return valueType == key; }

  static FutureTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<FutureTypeStorage>()) FutureTypeStorage(key);
  }

  Type valueType;
};

class FutureType : public Type::TypeBase<FutureType, Type, FutureTypeStorage> {
public:
  using Base::Base;

  static FutureType get(Type valueType) {
    return Base::get(valueType.getContext(), valueType);
  }

  Type getValueType() const { return getImpl()->valueType; }

  static constexpr StringLiteral name = "cira.future";
};

/// StreamType - A descriptor for a recurring memory pattern
struct StreamTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, int64_t, int64_t>;

  StreamTypeStorage(Type elementType, int64_t stride, int64_t offsetToNext)
      : elementType(elementType), stride(stride), offsetToNext(offsetToNext) {}

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == elementType && std::get<1>(key) == stride &&
           std::get<2>(key) == offsetToNext;
  }

  static StreamTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<StreamTypeStorage>())
        StreamTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  Type elementType;
  int64_t stride;
  int64_t offsetToNext;
};

class StreamType : public Type::TypeBase<StreamType, Type, StreamTypeStorage> {
public:
  using Base::Base;

  static StreamType get(Type elementType, int64_t stride = 0, int64_t offsetToNext = 0) {
    return Base::get(elementType.getContext(), elementType, stride, offsetToNext);
  }

  Type getElementType() const { return getImpl()->elementType; }
  int64_t getStride() const { return getImpl()->stride; }
  int64_t getOffsetToNext() const { return getImpl()->offsetToNext; }

  static constexpr StringLiteral name = "cira.stream";
};

} // namespace cira
} // namespace mlir

// Register types and operations
void RemoteMemDialect::initialize() {
    registerTypes();

    // Register Cira operations
    addOperations<
#define GET_OP_LIST
#include "Dialect/CiraOps.cpp.inc"
        >();
}

void RemoteMemDialect::registerTypes() {
    // Register the RemoteMemRefType
    addTypes<RemoteMemRefType>();

    // Register new CIRA types: Handle, Future, Stream
    addTypes<HandleType, FutureType, StreamType>();
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
    StringRef typeName;
    if (parser.parseKeyword(&typeName))
        return {};

    // Parse !cira.handle<T>
    if (typeName == "handle") {
        if (parser.parseLess())
            return {};
        Type elementType;
        if (parser.parseType(elementType))
            return {};
        if (parser.parseGreater())
            return {};
        return HandleType::get(elementType);
    }

    // Parse !cira.future<T>
    if (typeName == "future") {
        if (parser.parseLess())
            return {};
        Type valueType;
        if (parser.parseType(valueType))
            return {};
        if (parser.parseGreater())
            return {};
        return FutureType::get(valueType);
    }

    // Parse !cira.stream<T, stride=N, offset=M>
    if (typeName == "stream") {
        if (parser.parseLess())
            return {};
        Type elementType;
        if (parser.parseType(elementType))
            return {};

        int64_t stride = 0;
        int64_t offsetToNext = 0;

        // Parse optional stride and offset
        while (succeeded(parser.parseOptionalComma())) {
            StringRef paramName;
            if (parser.parseKeyword(&paramName))
                return {};
            if (parser.parseEqual())
                return {};
            int64_t value;
            if (parser.parseInteger(value))
                return {};

            if (paramName == "stride")
                stride = value;
            else if (paramName == "offset")
                offsetToNext = value;
        }

        if (parser.parseGreater())
            return {};
        return StreamType::get(elementType, stride, offsetToNext);
    }

    // Parse !cira.remotememref<T>
    if (typeName == "remotememref") {
        if (parser.parseLess())
            return {};
        Type elementType;
        if (parser.parseType(elementType))
            return {};
        if (parser.parseGreater())
            return {};
        return RemoteMemRefType::get(getContext(), elementType);
    }

    parser.emitError(parser.getNameLoc(), "unknown cira type: ") << typeName;
    return {};
}

void RemoteMemDialect::printType(Type type, DialectAsmPrinter &printer) const {
    llvm::TypeSwitch<Type>(type)
        .Case<HandleType>([&](HandleType handleType) {
            printer << "handle<" << handleType.getElementType() << ">";
        })
        .Case<FutureType>([&](FutureType futureType) {
            printer << "future<" << futureType.getValueType() << ">";
        })
        .Case<StreamType>([&](StreamType streamType) {
            printer << "stream<" << streamType.getElementType();
            if (streamType.getStride() != 0)
                printer << ", stride=" << streamType.getStride();
            if (streamType.getOffsetToNext() != 0)
                printer << ", offset=" << streamType.getOffsetToNext();
            printer << ">";
        })
        .Case<RemoteMemRefType>([&](RemoteMemRefType remoteType) {
            printer << "remotememref<" << remoteType.getElementType() << ">";
        })
        .Default([&](Type) { printer << "unknown_type"; });
}

// Include the generated dialect definitions
#include "Dialect/RemoteMemDialect.cpp.inc"