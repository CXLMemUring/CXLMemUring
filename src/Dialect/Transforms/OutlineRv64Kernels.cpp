//===- OutlineRv64Kernels.cpp - Deterministic RV64 device kernels --------===//

#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <string>

namespace mlir {
#define GEN_PASS_DEF_CIRAOUTLINERV64KERNELS
#include "Dialect/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::cira;

namespace {

struct SourceIdentity {
    std::string path;
    unsigned line = 0;
    unsigned column = 0;
};

struct PackedArgument {
    std::string name;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t alignment = 1;
    std::string type;
    Type mlirType;
    bool remoteOffset = false;
};

struct KernelRecord {
    OffloadRegionOp region;
    uint64_t kernelId = 0;
    std::string kernelIdText;
    std::string symbol;
    std::string source;
    SmallVector<int32_t> queryTags;
    SmallVector<PackedArgument> args;
    uint64_t argsSize = 0;
};

static std::optional<FileLineColLoc> findFileLocation(Location loc) {
    if (auto file = dyn_cast<FileLineColLoc>(loc))
        return file;
    if (auto fused = dyn_cast<FusedLoc>(loc))
        for (Location sub : fused.getLocations())
            if (auto file = findFileLocation(sub))
                return file;
    if (auto named = dyn_cast<NameLoc>(loc))
        return findFileLocation(named.getChildLoc());
    if (auto call = dyn_cast<CallSiteLoc>(loc))
        return findFileLocation(call.getCallee());
    return std::nullopt;
}

static SourceIdentity getSourceIdentity(Location loc) {
    SourceIdentity source{"<unknown>", 0, 0};
    if (auto file = findFileLocation(loc)) {
        SmallString<256> path(file->getFilename().getValue());
        llvm::sys::path::remove_dots(path, /*removeDotDot=*/true);
        source.path = path.str().str();
        source.line = file->getLine();
        source.column = file->getColumn();
    }
    return source;
}

static bool isRemotePointerType(Type type) {
    if (isa<HandleType, RemoteMemRefType>(type))
        return true;
    if (auto memref = dyn_cast<MemRefType>(type)) {
        Attribute space = memref.getMemorySpace();
        if (!space)
            return false;
        if (auto integer = dyn_cast<IntegerAttr>(space))
            return integer.getInt() != 0;
        return true;
    }
    if (auto pointer = dyn_cast<LLVM::LLVMPointerType>(type))
        return pointer.getAddressSpace() != 0;
    return false;
}

static bool isHostPointerType(Type type) {
    if (isa<MemRefType>(type))
        return !isRemotePointerType(type);
    if (auto pointer = dyn_cast<LLVM::LLVMPointerType>(type))
        return pointer.getAddressSpace() == 0;
    return false;
}

static uint64_t alignTo(uint64_t value, uint64_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

static FailureOr<PackedArgument> classifyArgument(Value operand, unsigned index) {
    PackedArgument arg;
    arg.name = "arg" + std::to_string(index);
    arg.mlirType = operand.getType();

    if (isRemotePointerType(arg.mlirType)) {
        arg.size = arg.alignment = 8;
        arg.type = "arena_offset";
        arg.remoteOffset = true;
        return arg;
    }
    if (isHostPointerType(arg.mlirType))
        return failure();
    if (arg.mlirType.isIndex()) {
        arg.size = arg.alignment = 8;
        arg.type = "index";
        return arg;
    }
    if (auto integer = dyn_cast<IntegerType>(arg.mlirType)) {
        if (integer.getWidth() == 0 || integer.getWidth() > 64)
            return failure();
        arg.size = std::max<uint64_t>(1, llvm::divideCeil(integer.getWidth(), 8u));
        arg.alignment = std::min<uint64_t>(llvm::PowerOf2Ceil(arg.size), 8);
        arg.type = "i" + std::to_string(integer.getWidth());
        return arg;
    }
    if (auto floating = dyn_cast<FloatType>(arg.mlirType)) {
        unsigned width = floating.getWidth();
        if (width != 16 && width != 32 && width != 64)
            return failure();
        arg.size = arg.alignment = width / 8;
        arg.type = "f" + std::to_string(width);
        return arg;
    }
    return failure();
}

static bool hasForbiddenEffect(Operation *op) {
    auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effectInterface)
        return false;
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectInterface.getEffects(effects);
    return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
        return isa<MemoryEffects::Write, MemoryEffects::Allocate, MemoryEffects::Free>(effect.getEffect());
    });
}

static LogicalResult validateRegion(OffloadRegionOp region) {
    for (Value operand : region.getOperands()) {
        if (isHostPointerType(operand.getType()))
            return region.emitError("cannot outline RV64 kernel: host pointer crosses device boundary");
        if (failed(classifyArgument(operand, 0)))
            return region.emitError("cannot outline RV64 kernel: unsupported boundary type ") << operand.getType();
    }

    WalkResult result = region.getBody().walk([&](Operation *op) -> WalkResult {
        if (auto call = dyn_cast<func::CallOp>(op)) {
            op->emitError("cannot outline RV64 kernel: unresolved call @")
                << call.getCallee() << " crosses device boundary";
            return WalkResult::interrupt();
        }
        if (hasForbiddenEffect(op)) {
            op->emitError("cannot outline RV64 kernel: write, allocation, or free is not read-only device code");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
}

static uint64_t stableKernelId(StringRef identity) {
    std::array<uint8_t, 32> digest = llvm::SHA256::hash(arrayRefFromStringRef(identity));
    uint64_t id = 0;
    for (unsigned i = 0; i != 8; ++i)
        id = (id << 8) | digest[i];
    return id;
}

static std::string hexId(uint64_t id) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << llvm::format_hex_no_prefix(id, 16, /*Upper=*/false);
    return text;
}

static ArrayAttr makeArgumentLayout(MLIRContext *context, ArrayRef<PackedArgument> args) {
    Builder builder(context);
    SmallVector<Attribute> entries;
    for (const PackedArgument &arg : args) {
        NamedAttrList fields;
        fields.set("name", builder.getStringAttr(arg.name));
        fields.set("offset", builder.getI64IntegerAttr(arg.offset));
        fields.set("type", builder.getStringAttr(arg.type));
        entries.push_back(builder.getDictionaryAttr(fields));
    }
    return builder.getArrayAttr(entries);
}

static SmallVector<int32_t> getQueryTags(OffloadRegionOp region) {
    SmallVector<int32_t> tags;
    if (auto dense = region->getAttrOfType<DenseI32ArrayAttr>("cira.query_tags"))
        tags.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
    else if (auto array = region->getAttrOfType<ArrayAttr>("cira.query_tags"))
        for (Attribute attr : array)
            if (auto integer = dyn_cast<IntegerAttr>(attr))
                tags.push_back(static_cast<int32_t>(integer.getInt()));
    return tags;
}

static void emitArgumentUnpack(OpBuilder &builder, KernelRecord &record, func::FuncOp kernel, IRMapping &mapping) {
    Location loc = record.region.getLoc();
    Value arena = kernel.getArgument(0);
    Value argsOffset = kernel.getArgument(1);
    Value argsAddress = builder.create<arith::AddIOp>(loc, arena, argsOffset);
    auto i64Type = builder.getI64Type();
    auto pointerType = LLVM::LLVMPointerType::get(builder.getContext());

    Block &body = record.region.getBody().front();
    for (auto [blockArg, packed] : llvm::zip(body.getArguments(), record.args)) {
        Value offset = builder.create<arith::ConstantIntOp>(loc, packed.offset, 64);
        Value slotAddress = builder.create<arith::AddIOp>(loc, argsAddress, offset);
        Value slotPointer = builder.create<LLVM::IntToPtrOp>(loc, pointerType, slotAddress);
        Value unpacked;
        if (packed.remoteOffset) {
            Value relative = builder.create<LLVM::LoadOp>(loc, i64Type, slotPointer);
            Value absolute = builder.create<arith::AddIOp>(loc, arena, relative);
            unpacked =
                builder.create<UnrealizedConversionCastOp>(loc, TypeRange{packed.mlirType}, absolute).getResult(0);
        } else if (packed.mlirType.isIndex()) {
            Value integer = builder.create<LLVM::LoadOp>(loc, i64Type, slotPointer);
            unpacked = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), integer);
        } else {
            unpacked = builder.create<LLVM::LoadOp>(loc, packed.mlirType, slotPointer);
        }
        mapping.map(blockArg, unpacked);
    }
}

static func::FuncOp getOrCreateDeviceHelper(ModuleOp module, StringRef name, FunctionType type) {
    if (auto helper = module.lookupSymbol<func::FuncOp>(name))
        return helper;
    OpBuilder declarationBuilder(module.getBodyRegion());
    declarationBuilder.setInsertionPointToStart(module.getBody());
    return declarationBuilder.create<func::FuncOp>(module.getLoc(), name, type);
}

static FailureOr<std::pair<uint64_t, uint64_t>> getLoadSizeAndAlignment(Type type) {
    uint64_t bytes = 0;
    if (auto integer = dyn_cast<IntegerType>(type)) {
        if (integer.getWidth() == 0 || integer.getWidth() > 64)
            return failure();
        bytes = std::max<uint64_t>(1, llvm::divideCeil(integer.getWidth(), 8u));
    } else if (auto floating = dyn_cast<FloatType>(type)) {
        unsigned width = floating.getWidth();
        if (width != 16 && width != 32 && width != 64)
            return failure();
        bytes = width / 8;
    } else if (type.isIndex() || isa<LLVM::LLVMPointerType>(type)) {
        bytes = 8;
    } else {
        return failure();
    }
    return std::make_pair(bytes, std::min<uint64_t>(llvm::PowerOf2Ceil(bytes), 8));
}

static FailureOr<Value> getArenaRelativeLoadAddress(OpBuilder &builder, Operation *operation, Value arena,
                                                    uint64_t elementBytes) {
    Location loc = operation->getLoc();
    Value absoluteAddress;
    if (auto load = dyn_cast<LLVM::LoadOp>(operation)) {
        absoluteAddress = builder.create<LLVM::PtrToIntOp>(loc, builder.getI64Type(), load.getAddr());
    } else if (auto load = dyn_cast<memref::LoadOp>(operation)) {
        auto memrefType = dyn_cast<MemRefType>(load.getMemRefType());
        SmallVector<int64_t> strides;
        int64_t staticOffset = 0;
        if (!memrefType || failed(memrefType.getStridesAndOffset(strides, staticOffset)) ||
            ShapedType::isDynamic(staticOffset) || llvm::is_contained(strides, ShapedType::kDynamic) ||
            strides.size() != load.getIndices().size()) {
            operation->emitError("checked RV64 load requires a statically strided memref address");
            return failure();
        }
        Value pointerIndex = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, load.getMemref());
        absoluteAddress = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), pointerIndex);
        int64_t staticByteOffset = staticOffset * static_cast<int64_t>(elementBytes);
        if (staticByteOffset) {
            Value offset = builder.create<arith::ConstantIntOp>(loc, staticByteOffset, 64);
            absoluteAddress = builder.create<arith::AddIOp>(loc, absoluteAddress, offset);
        }
        for (auto [index, stride] : llvm::zip(load.getIndices(), strides)) {
            Value index64 = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), index);
            Value byteStride =
                builder.create<arith::ConstantIntOp>(loc, stride * static_cast<int64_t>(elementBytes), 64);
            Value dynamicOffset = builder.create<arith::MulIOp>(loc, index64, byteStride);
            absoluteAddress = builder.create<arith::AddIOp>(loc, absoluteAddress, dynamicOffset);
        }
    } else {
        operation->emitError("checked RV64 load is not a supported load operation");
        return failure();
    }
    return builder.create<arith::SubIOp>(loc, absoluteAddress, arena).getResult();
}

static Value convertCheckedLoadResult(OpBuilder &builder, Location loc, Value checked, Type resultType) {
    if (resultType == builder.getI64Type())
        return checked;
    if (auto integer = dyn_cast<IntegerType>(resultType))
        return builder.create<arith::TruncIOp>(loc, integer, checked);
    if (auto floating = dyn_cast<FloatType>(resultType)) {
        Type bitsType = builder.getIntegerType(floating.getWidth());
        Value bits = floating.getWidth() == 64 ? checked : builder.create<arith::TruncIOp>(loc, bitsType, checked);
        return builder.create<arith::BitcastOp>(loc, floating, bits);
    }
    if (resultType.isIndex())
        return builder.create<arith::IndexCastOp>(loc, resultType, checked);
    if (isa<LLVM::LLVMPointerType>(resultType))
        return builder.create<LLVM::IntToPtrOp>(loc, resultType, checked);
    llvm_unreachable("load type was validated before conversion");
}

static LogicalResult instrumentDeviceLoadsAndInstalls(ModuleOp module, func::FuncOp kernel) {
    constexpr StringLiteral checkedLoadName = "cira_device_checked_load";
    constexpr StringLiteral installName = "cira_device_install_cacheline";
    SmallVector<Operation *> loads;
    bool hasPreparedRoot = false;
    kernel.walk([&](Operation *op) {
        if (!isa<LLVM::LoadOp, memref::LoadOp>(op))
            return;
        loads.push_back(op);
        hasPreparedRoot |= op->hasAttr("cira.prepared_root");
    });
    if (loads.empty())
        return success();

    {
        OpBuilder declarationBuilder(module.getBodyRegion());
        Type i64 = declarationBuilder.getI64Type();
        Type i32 = declarationBuilder.getI32Type();
        getOrCreateDeviceHelper(module, checkedLoadName, declarationBuilder.getFunctionType({i64, i32, i32}, {i64}));
        if (hasPreparedRoot)
            getOrCreateDeviceHelper(module, installName, declarationBuilder.getFunctionType({i64}, {}));
    }

    Value arena = kernel.getArgument(0);
    for (Operation *load : loads) {
        Value loadResult = load->getResult(0);
        FailureOr<std::pair<uint64_t, uint64_t>> layout = getLoadSizeAndAlignment(loadResult.getType());
        if (failed(layout))
            return load->emitError("checked RV64 load has unsupported result type ") << loadResult.getType();

        OpBuilder builder(load);
        Location loc = load->getLoc();
        FailureOr<Value> arenaOffset = getArenaRelativeLoadAddress(builder, load, arena, layout->first);
        if (failed(arenaOffset))
            return failure();
        Value bytes = builder.create<arith::ConstantIntOp>(loc, layout->first, 32);
        Value alignment = builder.create<arith::ConstantIntOp>(loc, layout->second, 32);
        auto checkedCall = builder.create<func::CallOp>(loc, checkedLoadName, TypeRange{builder.getI64Type()},
                                                        ValueRange{*arenaOffset, bytes, alignment});
        Value replacement = convertCheckedLoadResult(builder, loc, checkedCall.getResult(0), loadResult.getType());

        if (load->hasAttr("cira.prepared_root")) {
            Value lineMask = builder.create<arith::ConstantIntOp>(loc, -64, 64);
            Value lineOffset = builder.create<arith::AndIOp>(loc, *arenaOffset, lineMask);
            builder.create<func::CallOp>(loc, installName, TypeRange{}, ValueRange{lineOffset});
        }
        loadResult.replaceAllUsesWith(replacement);
        load->erase();
    }
    return success();
}

static func::FuncOp outlineKernel(ModuleOp module, KernelRecord &record) {
    MLIRContext *context = module.getContext();
    OpBuilder moduleBuilder(module.getBodyRegion());
    moduleBuilder.setInsertionPointToEnd(module.getBody());
    Type i64 = moduleBuilder.getI64Type();
    auto functionType = moduleBuilder.getFunctionType({i64, i64}, {});
    auto kernel = moduleBuilder.create<func::FuncOp>(record.region.getLoc(), record.symbol, functionType);
    kernel.setPrivate();
    kernel->setAttr("cira.rv64_kernel", UnitAttr::get(context));
    kernel->setAttr("cira.read_only", UnitAttr::get(context));
    kernel->setAttr("cira.kernel_id", moduleBuilder.getI64IntegerAttr(record.kernelId));
    kernel->setAttr("cira.args_size", moduleBuilder.getI64IntegerAttr(record.argsSize));
    kernel->setAttr("cira.arg_layout", makeArgumentLayout(context, record.args));
    kernel->setAttr("target-cpu", moduleBuilder.getStringAttr("generic-rv64"));

    Block *entry = kernel.addEntryBlock();
    OpBuilder bodyBuilder(entry, entry->begin());
    IRMapping mapping;
    emitArgumentUnpack(bodyBuilder, record, kernel, mapping);
    for (Operation &op : record.region.getBody().front())
        if (!isa<YieldOp>(op))
            bodyBuilder.clone(op, mapping);
    bodyBuilder.create<func::ReturnOp>(record.region.getLoc());
    if (failed(instrumentDeviceLoadsAndInstalls(module, kernel)))
        return {};
    return kernel;
}

static LogicalResult writeManifest(StringRef path, ArrayRef<KernelRecord> records, Operation *anchor) {
    std::error_code error;
    llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_Text);
    if (error)
        return anchor->emitError("cannot write RV64 kernel manifest '") << path << "': " << error.message();

    {
        llvm::json::OStream json(stream, 2);
        json.object([&] {
            json.attribute("abi_version", 2);
            json.attributeArray("kernels", [&] {
                for (const KernelRecord &record : records) {
                    json.object([&] {
                        json.attribute("kernel_id", "0x" + record.kernelIdText);
                        json.attribute("symbol", record.symbol);
                        json.attribute("source", record.source);
                        json.attributeArray("query_tags", [&] {
                            for (int32_t tag : record.queryTags)
                                json.value(tag);
                        });
                        json.attribute("args_size", static_cast<int64_t>(record.argsSize));
                        json.attributeArray("args", [&] {
                            for (const PackedArgument &arg : record.args) {
                                json.object([&] {
                                    json.attribute("name", arg.name);
                                    json.attribute("offset", static_cast<int64_t>(arg.offset));
                                    json.attribute("type", arg.type);
                                });
                            }
                        });
                        json.attribute("read_only", true);
                    });
                }
            });
        });
    }
    stream << '\n';
    return success();
}

class CiraOutlineRv64KernelsPass : public impl::CiraOutlineRv64KernelsBase<CiraOutlineRv64KernelsPass> {
public:
    using impl::CiraOutlineRv64KernelsBase<CiraOutlineRv64KernelsPass>::CiraOutlineRv64KernelsBase;

    void runOnOperation() override {
        ModuleOp module = getOperation();
        if (manifest.empty()) {
            module.emitError("cira-outline-rv64-kernels requires a non-empty manifest path");
            return signalPassFailure();
        }

        SmallVector<OffloadRegionOp> regions;
        module.walk([&](OffloadRegionOp region) { regions.push_back(region); });
        SmallVector<KernelRecord, 0> records;
        llvm::DenseMap<uint64_t, Operation *> identities;
        unsigned fallbackOrdinal = 0;

        for (OffloadRegionOp region : regions) {
            if (failed(validateRegion(region))) {
                signalPassFailure();
                return;
            }

            auto function = region->getParentOfType<func::FuncOp>();
            SourceIdentity source = getSourceIdentity(region.getLoc());
            uint64_t ordinal = fallbackOrdinal++;
            if (auto attr = region->getAttrOfType<IntegerAttr>("cira.region_id"))
                ordinal = attr.getValue().getZExtValue();
            std::string identity = (Twine(moduleId) + " | " + source.path + " | " + function.getSymName() + " | " +
                                    Twine(source.line) + ":" + Twine(source.column) + " | " + Twine(ordinal))
                                       .str();

            KernelRecord record;
            record.region = region;
            record.kernelId = stableKernelId(identity);
            record.kernelIdText = hexId(record.kernelId);
            record.symbol = "__cira_rv64_" + record.kernelIdText;
            record.source = (Twine(source.path) + ":" + Twine(source.line) + ":" + Twine(source.column)).str();
            record.queryTags = getQueryTags(region);

            if (Operation *collision = identities.lookup(record.kernelId)) {
                region.emitError("RV64 kernel ID collision with region at ") << collision->getLoc();
                signalPassFailure();
                return;
            }
            if (module.lookupSymbol(record.symbol)) {
                region.emitError("RV64 kernel symbol already exists: ") << record.symbol;
                signalPassFailure();
                return;
            }
            identities[record.kernelId] = region;

            uint64_t offset = 0;
            for (auto [index, operand] : llvm::enumerate(region.getOperands())) {
                FailureOr<PackedArgument> arg = classifyArgument(operand, index);
                if (failed(arg)) {
                    region.emitError("cannot outline RV64 kernel: unsupported boundary type ") << operand.getType();
                    signalPassFailure();
                    return;
                }
                offset = alignTo(offset, arg->alignment);
                arg->offset = offset;
                offset += arg->size;
                record.args.push_back(std::move(*arg));
            }
            record.argsSize = alignTo(offset, 8);
            records.push_back(std::move(record));
        }

        llvm::sort(records,
                   [](const KernelRecord &left, const KernelRecord &right) { return left.kernelId < right.kernelId; });
        for (KernelRecord &record : records) {
            ArrayAttr layout = makeArgumentLayout(&getContext(), record.args);
            record.region->setAttr("cira.kernel_id",
                                   IntegerAttr::get(IntegerType::get(&getContext(), 64), APInt(64, record.kernelId)));
            record.region->setAttr("cira.args_size",
                                   IntegerAttr::get(IntegerType::get(&getContext(), 64), record.argsSize));
            record.region->setAttr("cira.arg_layout", layout);
            if (!outlineKernel(module, record)) {
                signalPassFailure();
                return;
            }
        }
        if (failed(writeManifest(manifest, records, module)))
            signalPassFailure();
    }
};

} // namespace
