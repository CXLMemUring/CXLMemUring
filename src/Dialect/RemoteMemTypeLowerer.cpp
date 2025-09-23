
#include "Dialect/RemoteMemDialect.h"
#include "Dialect/RemoteMemRef.h"
#include "Dialect/RemoteMemTypeLower.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::cira;

RemoteMemTypeLowerer::RemoteMemTypeLowerer(MLIRContext *ctx, DataLayoutAnalysis const *analysis)
    : RemoteMemTypeLowerer(ctx, LowerToLLVMOptions(ctx), analysis) {}

RemoteMemTypeLowerer::RemoteMemTypeLowerer(MLIRContext *ctx, const LowerToLLVMOptions &options,
                                           DataLayoutAnalysis const *analysis)
    : options(options), dataLayoutAnalysis(analysis) {

    // 获取RemoteMemDialect实例
    rmemDialect = ctx->getOrLoadDialect<RemoteMemDialect>();
    assert(rmemDialect && "RemoteMemDialect must be loaded");

    // 添加转换规则
    addConversion([&](RemoteMemRefType type) -> Type { return convertRemoteMemRefToPtr(type); });

    // 添加其他类型转换规则
    addConversion([&](ComplexType type) -> Type { return convertComplexType(type); });

    addConversion([&](IndexType type) -> Type { return convertIndexType(type); });

    addConversion([&](FloatType type) -> Type { return convertFloatType(type); });

    addConversion([&](IntegerType type) -> Type { return convertIntegerType(type); });

    addConversion([&](VectorType type) -> Type { return convertVectorType(type); });

    // 标记已知可以传递的类型
    addConversion([](LLVM::LLVMStructType type) -> Type { return type; });
    addConversion([](LLVM::LLVMPointerType type) -> Type { return type; });
    addConversion([](LLVM::LLVMArrayType type) -> Type { return type; });
    addConversion([](LLVM::LLVMFunctionType type) -> Type { return type; });

    // 如果需要，添加更多类型转换规则
}

MLIRContext &RemoteMemTypeLowerer::getContext() { return *rmemDialect->getContext(); }

Type RemoteMemTypeLowerer::getIndexType() { return IntegerType::get(&getContext(), options.getIndexBitwidth()); }

unsigned RemoteMemTypeLowerer::getPointerBitwidth(unsigned addressSpace) {
    return options.dataLayout.getPointerSizeInBits(addressSpace);
}

Type RemoteMemTypeLowerer::convertIndexType(IndexType type) { return getIndexType(); }

Type RemoteMemTypeLowerer::convertIntegerType(IntegerType type) {
    return IntegerType::get(&getContext(), type.getWidth());
}

Type RemoteMemTypeLowerer::convertFloatType(FloatType type) { return type; }

Type RemoteMemTypeLowerer::convertComplexType(ComplexType type) {
    Type elementType = convertType(type.getElementType());
    if (!elementType)
        return {};
    return LLVM::LLVMStructType::getLiteral(&getContext(), {elementType, elementType});
}

Type RemoteMemTypeLowerer::convertVectorType(VectorType type) {
    // 只支持1D向量
    if (type.getRank() != 1)
        return {};

    Type elementType = convertType(type.getElementType());
    if (!elementType)
        return {};

    return LLVM::getVectorType(elementType, type.getDimSize(0));
}

Type RemoteMemTypeLowerer::convertRemoteMemRefToPtr(RemoteMemRefType type) {
    Type elementType = type.getElementType();

    // 如果已经是LLVM指针类型，则保持不变
    if (auto ptrType = llvm::dyn_cast<LLVM::LLVMPointerType>(elementType)) {
        return ptrType;
    }

    // 否则，转换为LLVM指针类型
    if (auto memRefType = llvm::dyn_cast<MemRefType>(elementType)) {
        Type convertedType = convertType(memRefType.getElementType());
        if (!convertedType)
            return {};
        return LLVM::LLVMPointerType::get(&getContext());
    }

    // 如果是未排序的MemRef，则转换为void指针
    if (llvm::isa<UnrankedMemRefType>(elementType)) {
        return LLVM::LLVMPointerType::get(&getContext());
    }

    return {};
}

Type RemoteMemTypeLowerer::convertRemoteMemRefToMemRefDesc(RemoteMemRefType type) {
    // 不完整的实现，根据需要完善
    return convertRemoteMemRefToPtr(type);
}

SmallVector<Type, 2> RemoteMemTypeLowerer::getUnrankedMemRefDescriptorFields() {
    SmallVector<Type, 2> result;
    result.push_back(getIndexType()); // Rank
    result.push_back(LLVM::LLVMPointerType::get(&getContext())); // 指向数据的指针
    return result;
}

Type RemoteMemTypeLowerer::convertRemoteMemRefToUnrankedDesc(RemoteMemRefType type) {
    auto fields = getUnrankedMemRefDescriptorFields();
    return LLVM::LLVMStructType::getLiteral(&getContext(), fields);
}

SmallVector<Type, 5> RemoteMemTypeLowerer::getMemRefDescriptorFields(MemRefType type, bool unpackAggregates) {
    // 基本实现，根据需要完善
    SmallVector<Type, 5> result;
    auto elementType = convertType(type.getElementType());

    // 分配指针
    result.push_back(LLVM::LLVMPointerType::get(&getContext()));
    // 对齐指针
    result.push_back(LLVM::LLVMPointerType::get(&getContext()));
    // 偏移
    result.push_back(getIndexType());

    auto rank = type.getRank();
    if (unpackAggregates) {
        // 尺寸
        for (unsigned i = 0; i < rank; ++i)
            result.push_back(getIndexType());
        // 步幅
        for (unsigned i = 0; i < rank; ++i)
            result.push_back(getIndexType());
    } else {
        // 尺寸数组
        result.push_back(LLVM::LLVMArrayType::get(&getContext(), getIndexType(), rank));
        // 步幅数组
        result.push_back(LLVM::LLVMArrayType::get(&getContext(), getIndexType(), rank));
    }

    return result;
}

unsigned RemoteMemTypeLowerer::getMemRefDescriptorSize(MemRefType type, const DataLayout &layout) {
    // 简单实现
    return 8 + 8 + 8 + type.getRank() * 16; // 根据需要调整
}

bool RemoteMemTypeLowerer::canConvertToBarePtr(BaseMemRefType type) {
    if (auto memrefTy = llvm::dyn_cast<MemRefType>(type)) {
        return memrefTy.hasStaticShape() && memrefTy.getLayout().isIdentity() && memrefTy.getMemorySpace() == 0;
    }
    return false;
}

Value RemoteMemTypeLowerer::promoteOneMemRefDescriptor(Location loc, Value operand, OpBuilder &builder) {
    // 待实现
    return operand;
}

SmallVector<Value, 4> RemoteMemTypeLowerer::promoteOperands(Location loc, ValueRange opOperands, ValueRange operands,
                                                            OpBuilder &builder) {
    // 待实现
    SmallVector<Value, 4> promotedOperands;
    for (auto operand : operands) {
        promotedOperands.push_back(operand);
    }
    return promotedOperands;
}

Type RemoteMemTypeLowerer::convertFunctionSignature(FunctionType funcTy, SignatureConversion &result) {
    // 待实现
    return convertFunctionType(funcTy);
}

Type RemoteMemTypeLowerer::convertCallingConventionType(Type type) {
    // 待实现
    return convertType(type);
}

LogicalResult RemoteMemTypeLowerer::structFuncArgTypeConverter(Type type, SmallVector<Type> &result) {
    // 待实现
    result.push_back(convertType(type));
    return success();
}

Type RemoteMemTypeLowerer::packFunctionResults(TypeRange types) {
    // 待实现
    if (types.empty())
        return LLVM::LLVMVoidType::get(&getContext());
    if (types.size() == 1)
        return convertType(types[0]);

    SmallVector<Type, 4> resultTypes;
    for (auto type : types) {
        resultTypes.push_back(convertType(type));
    }
    return LLVM::LLVMStructType::getLiteral(&getContext(), resultTypes);
}

Type RemoteMemTypeLowerer::convertFunctionType(FunctionType funcTy) {
    // 转换输入类型
    SmallVector<Type, 4> inputTypes;
    for (auto input : funcTy.getInputs()) {
        auto converted = convertType(input);
        if (!converted)
            return {};
        inputTypes.push_back(converted);
    }

    // 转换结果类型
    SmallVector<Type, 4> resultTypes;
    for (auto result : funcTy.getResults()) {
        auto converted = convertType(result);
        if (!converted)
            return {};
        resultTypes.push_back(converted);
    }

    // 创建LLVM函数类型
    Type resultType;
    if (resultTypes.empty())
        resultType = LLVM::LLVMVoidType::get(&getContext());
    else if (resultTypes.size() == 1)
        resultType = resultTypes[0];
    else
        resultType = LLVM::LLVMStructType::getLiteral(&getContext(), resultTypes);

    return LLVM::LLVMFunctionType::get(resultType, inputTypes, false);
}

LogicalResult RemoteMemTypeLowerer::convertStructType(LLVM::LLVMStructType type, SmallVectorImpl<Type> &results,
                                                      ArrayRef<Type> callStack) {
    // 待实现
    results.push_back(type);
    return success();
}