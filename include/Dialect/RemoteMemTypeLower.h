//
// Created by yangyw on 10/10/24.
//

#ifndef CIRA_REMOTEMEMTYPELOWER_H
#define CIRA_REMOTEMEMTYPELOWER_H
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class DataLayoutAnalysis;
class LowerToLLVMOptions;
class LLVMTypeConverter;

namespace cira {
class RemoteMemDialect;
class RemoteMemRefType;
class RemoteMemTypeLowerer : public TypeConverter {
public:
    using TypeConverter::convertType;
    RemoteMemTypeLowerer(MLIRContext *ctx,  DataLayoutAnalysis const*analysis = nullptr);
    RemoteMemTypeLowerer(MLIRContext *ctx, const LowerToLLVMOptions &options,
                         DataLayoutAnalysis const*analysis = nullptr);

    MLIRContext &getContext();
    RemoteMemDialect *getDialect() { return rmemDialect; }

    const LowerToLLVMOptions &getOptions() const { return options; }

    /// Returns the data layout to use during and after conversion.
    const llvm::DataLayout &getDataLayout() { return options.dataLayout; }

    /// Returns the data layout analysis to query during conversion.
    const DataLayoutAnalysis *getDataLayoutAnalysis() const { return dataLayoutAnalysis; }

    /// Gets the LLVM representation of the index type. The returned type is an
    /// integer type with the size configured for this type converter.
    Type getIndexType();

    /// Gets the bitwidth of the index type when converted to LLVM.
    unsigned getIndexTypeBitwidth() { return options.getIndexBitwidth(); }

    /// Gets the pointer bitwidth.
    unsigned getPointerBitwidth(unsigned addressSpace = 0);

    /// Returns the size of the memref descriptor object in bytes.
    unsigned getMemRefDescriptorSize(MemRefType type, const DataLayout &layout);

    /// Check if a memref type can be converted to a bare pointer.
    static bool canConvertToBarePtr(BaseMemRefType type);

    /// Promote the LLVM representation of all operands including promoting MemRef
    /// descriptors to stack and use pointers to struct to avoid the complexity
    /// of the platform-specific C/C++ ABI lowering related to struct argument
    /// passing.
    SmallVector<Value, 4> promoteOperands(Location loc, ValueRange opOperands, ValueRange operands, OpBuilder &builder);

    /// Promote the LLVM struct representation of one MemRef descriptor to stack
    /// and use pointer to struct to avoid the complexity of the platform-specific
    /// C/C++ ABI lowering related to struct argument passing.
    Value promoteOneMemRefDescriptor(Location loc, Value operand, OpBuilder &builder);

    Type convertFunctionSignature(FunctionType funcTy, SignatureConversion &result);
    Type convertCallingConventionType(Type type);
    LogicalResult structFuncArgTypeConverter(Type type, SmallVector<Type> &result);
    Type packFunctionResults(TypeRange types);

protected:
    RemoteMemDialect *rmemDialect;

private:
    Type convertFunctionType(FunctionType funcTy);

    /// Convert the index type.  Uses llvmModule data layout to create an integer
    /// of the pointer bitwidth.
    Type convertIndexType(IndexType type);

    /// Convert an integer type `i*` to `!llvm<"i*">`.
    Type convertIntegerType(IntegerType type);

    /// Convert a floating point type: `f16` to `f16`, `f32` to
    /// `f32` and `f64` to `f64`.  `bf16` is not supported
    /// by LLVM.
    Type convertFloatType(FloatType type);

    /// Convert complex number type: `complex<f16>` to `!llvm<"{ half, half }">`,
    /// `complex<f32>` to `!llvm<"{ float, float }">`, and `complex<f64>` to
    /// `!llvm<"{ double, double }">`. `complex<bf16>` is not supported.
    Type convertComplexType(ComplexType type);

    /// Convert a 1D vector type into an LLVM vector type.
    Type convertVectorType(VectorType type);

    Type convertRemoteMemRefToPtr(RemoteMemRefType type);

    Type convertRemoteMemRefToMemRefDesc(RemoteMemRefType type);

    SmallVector<Type, 2> getUnrankedMemRefDescriptorFields();
    Type convertRemoteMemRefToUnrankedDesc(RemoteMemRefType type);

    /// Convert a memref type into a list of LLVM IR types that will form the
    /// memref descriptor. If `unpackAggregates` is true the `sizes` and `strides`
    /// arrays in the descriptors are unpacked to individual index-typed elements,
    /// else they are are kept as rank-sized arrays of index type. In particular,
    /// the list will contain:
    /// - two virtual pointers to the memref element type, followed by
    /// - an index-typed offset, followed by
    /// - (if unpackAggregates = true)
    ///    - one index-typed size per dimension of the memref, followed by
    ///    - one index-typed stride per dimension of the memref.
    /// - (if unpackArrregates = false)
    ///   - one rank-sized array of index-type for the size of each dimension
    ///   - one rank-sized array of index-type for the stride of each dimension
    ///
    /// For example, memref<?x?xf32> is converted to the following list:
    /// - `!llvm<"float*">` (allocated pointer),
    /// - `!llvm<"float*">` (aligned pointer),
    /// - `i64` (offset),
    /// - `i64`, `i64` (sizes),
    /// - `i64`, `i64` (strides).
    /// These types can be recomposed to a memref descriptor struct.
    SmallVector<Type, 5> getMemRefDescriptorFields(MemRefType type, bool unpackAggregates);

    LogicalResult convertStructType(LLVM::LLVMStructType type, SmallVectorImpl<Type> &results,
                                    ArrayRef<Type> callStack);

    /// Options for customizing the llvm lowering.
    LowerToLLVMOptions options;

    /// Data layout analysis mapping scopes to layouts active in them.
    const DataLayoutAnalysis *dataLayoutAnalysis;
};
} // namespace cira
} // namespace mlir
#endif // CIRA_REMOTEMEMTYPELOWER_H
