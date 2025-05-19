//
// Created by yangyw on 8/4/24.
//

#ifndef CIRA_REMOTEMEM_H
#define CIRA_REMOTEMEM_H

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/raw_ostream.h"

#include "Conversion/CIRA.h"
#include "Dialect/RemoteMemTypeLower.h"
#include "Dialect/RemoteMemDialect.h"
#include "Dialect/RemoteMemRef.h"
#include "Dialect/OffloadOp.h"

namespace mlir {
class Value;
class Type;
namespace cira {
class RemoteMemDialect;
class RemoteMemRefType;

class RemoteMemLoweringPattern : public ConversionPattern {
public:
    RemoteMemLoweringPattern(StringRef rootOpName, MLIRContext *context, cira::RemoteMemTypeLowerer &typeConverter,
                             PatternBenefit benefit = 1);

protected:
    cira::RemoteMemDialect &getDialect() const;
    cira::RemoteMemTypeLowerer *getTypeConverter() const;

    //===---------------------------------------------------===//
    // Remote Mem common rountines
    //===---------------------------------------------------===//

    // Materialize a disagg virtual address to local laddress
    //    Value materializeDisaggVirtualAddress(PatternRewriter &rewriter, Operation *op, Value dvaddr, Type relType,
    //    unsigned accessType = rmem::ACCESS) const;

    // Materialize a disagg virtual address to local laddress
    // use new_runtime
    //    Value newMatDisaggVirtualAddress(PatternRewriter &rewriter, Operation *op, Value dvaddr, Type relType,
    //    rmem::Cache *cache, unsigned accessType = rmem::ACCESS) const;

    //===---------------------------------------------------===//
    // Original LLVM common rountines
    //===---------------------------------------------------===//

    /// Gets the MLIR type wrapping the LLVM integer type whose bit width is
    /// defined by the used type converter.
    Type getIndexType() const;

    /// Gets the MLIR type wrapping the LLVM integer type whose bit width
    /// corresponds to that of a LLVM pointer type.
    Type getIntPtrType(unsigned addressSpace = 0) const;

    /// Gets the MLIR type wrapping the LLVM void type.
    Type getVoidType() const;

    /// Get the MLIR type wrapping the LLVM i8* type.
    Type getVoidPtrType() const;

    /// Create a constant Op producing a value of `resultType` from an index-typed
    /// integer attribute.
    static Value createIndexAttrConstant(OpBuilder &builder, Location loc, Type resultType, int64_t value);

    /// Create an LLVM dialect operation defining the given index constant.
    Value createIndexConstant(ConversionPatternRewriter &builder, Location loc, uint64_t value) const;

    // This is a strided getElementPtr variant that linearizes subscripts as:
    //   `base_offset + index_0 * stride_0 + ... + index_n * stride_n`.
    Value getStridedElementPtr(Location loc, MemRefType type, Value memRefDesc, ValueRange indices,
                               ConversionPatternRewriter &rewriter) const;

    // If is memrefType,
    /// Returns if the given memref has identity maps and the element type is
    /// convertible to LLVM.
    bool isConvertibleAndHasIdentityMaps(MemRefType type) const;

    /// Returns the type of a pointer to an element of the memref.
    Type getElementPtrType(MemRefType type) const;

    /// Computes sizes, strides and buffer size in bytes of `memRefType` with
    /// identity layout. Emits constant ops for the static sizes of `memRefType`,
    /// and uses `dynamicSizes` for the others. Emits instructions to compute
    /// strides and buffer size from these sizes.
    ///
    /// For example, memref<4x?xf32> emits:
    /// `sizes[0]`   = llvm.mlir.constant(4 : index) : i64
    /// `sizes[1]`   = `dynamicSizes[0]`
    /// `strides[1]` = llvm.mlir.constant(1 : index) : i64
    /// `strides[0]` = `sizes[0]`
    /// %size        = llvm.mul `sizes[0]`, `sizes[1]` : i64
    /// %nullptr     = llvm.mlir.null : !llvm.ptr<f32>
    /// %gep         = llvm.getelementptr %nullptr[%size]
    ///                  : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    /// `sizeBytes`  = llvm.ptrtoint %gep : !llvm.ptr<f32> to i64
    void getMemRefDescriptorSizes(Location loc, MemRefType memRefType, ValueRange dynamicSizes,
                                 ConversionPatternRewriter &rewriter, SmallVectorImpl<Value> &sizes,
                                 SmallVectorImpl<Value> &strides, Value &sizeBytes) const;

    /// Computes the size of type in bytes.
    Value getSizeInBytes(Location loc, Type type, ConversionPatternRewriter &rewriter) const;

    /// Computes total number of elements for the given shape.
    Value getNumElements(Location loc, ArrayRef<Value> shape, ConversionPatternRewriter &rewriter) const;

    /// Creates and populates a canonical memref descriptor struct.
    MemRefDescriptor createMemRefDescriptor(Location loc, MemRefType memRefType, Value allocatedPtr, Value alignedPtr,
                                         ArrayRef<Value> sizes, ArrayRef<Value> strides,
                                         ConversionPatternRewriter &rewriter) const;

    // block_base = lbase + (index % num_blocks) * block_size
    //    Value getBlockAddr(ModuleOp mop, Value curIndex, rmem::LocalCache &cache, Location loc,
    //    ConversionPatternRewriter &rewriter) const;
};

template <typename SourceOp> class RemoteMemOpLoweringPattern : public RemoteMemLoweringPattern {
public:
    using OpAdaptor = typename SourceOp::Adaptor;
    explicit RemoteMemOpLoweringPattern(cira::RemoteMemTypeLowerer typeConverter, MLIRContext *ctx,
                                     PatternBenefit bft = 1)
        : RemoteMemLoweringPattern(SourceOp::getOperationName(), ctx, typeConverter, bft) {}
    void rewrite(Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final {
        rewrite(cast<SourceOp>(op), OpAdaptor(operands, op->getAttrDictionary()), rewriter);
    }

    LogicalResult match(Operation *op) const final { return match(cast<SourceOp>(op)); }

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const final {
        return matchAndRewrite(cast<SourceOp>(op), OpAdaptor(operands, op->getAttrDictionary()), rewriter);
    }

    // match and rewrite on the concrete source op, must be override by the derived class
    virtual void rewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
        llvm_unreachable("must overwrite rewrite or matchAndRewirte");
    }

    virtual LogicalResult match(SourceOp op) const { llvm_unreachable("must overwrite match or matchAndRewrite"); }

    virtual LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
        if (failed(match(op)))
            return mlir::failure();
        rewrite(op, adaptor, rewriter);
        return success();
    }

    Value getTypeSize(Type t, ConversionPatternRewriter &rewriter);

private:
    using RemoteMemLoweringPattern::matchAndRewrite;
};

} // namespace cira
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/RemoteMem.h.inc"

#endif // CIRA_REMOTEMEM_H