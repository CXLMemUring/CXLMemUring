//
// Created by yangyw on 9/19/24.
//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ScopedPrinter.h"

#include "Dialect/ReadFromProfile.h"
#include "Dialect/RemoteMem.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>
#include <string>

using namespace mlir;

static void loadProfilingFromFile(std::string fileName, std::vector<ProfilingResult> &docs) {
    auto reader = llvm::MemoryBuffer::getFile(fileName);

    llvm::yaml::Input yin(**reader);
    yin >> docs;

    if (yin.error())
        return;
}

Type typeAttrArrayToType(Attribute attr) {
    auto typeAttr = llvm::dyn_cast<ArrayAttr>(attr)[0];
    return llvm::dyn_cast<TypeAttr>(typeAttr).getValue();
}

size_t transformTypeSize(Type type) {
    unsigned factor = 512 * 1024;
    if (isa<LLVM::LLVMPointerType>(type)) {
        // LLVMPointerType no longer has getElementType in opaque pointer mode
        return factor; // Return default size for pointers
    } else if (isa<LLVM::LLVMStructType>(type)) {
        auto structType = cast<LLVM::LLVMStructType>(type);
        for (auto elementType : structType.getBody()) {
            return transformTypeSize(elementType);
        }
    } else if (isa<IntegerType>(type) || isa<FloatType>(type)) {
        return type.getIntOrFloatBitWidth() / 8 * factor;
    }
}

Type transformStruct(MLIRContext *context, Type type, int cacheId = 0) {
    if (isa<LLVM::LLVMPointerType>(type)) {
        // Opaque pointers don't have element types - wrap in RemoteMemRef
        auto ptrType = LLVM::LLVMPointerType::get(context);
        return cira::RemoteMemRefType::get(ptrType, cacheId);
    } else if (isa<LLVM::LLVMStructType>(type)) {
        auto structType = cast<LLVM::LLVMStructType>(type);
        std::vector<Type> newElementTypes;
        for (auto elementType : structType.getBody()) {
            newElementTypes.push_back(transformStruct(context, elementType, cacheId));
        }
        return LLVM::LLVMStructType::getLiteral(context, newElementTypes);
    } else
        return type;
}

unsigned walkRemoteType(Type type) {
    if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type))
        // Opaque pointers - return default depth
        return 1;
    else if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
        for (auto elementType : structType.getBody())
            return walkRemoteType(elementType);
    } else if (auto rmrefType = dyn_cast<cira::RemoteMemRefType>(type)) {
        return rmrefType.getCanRemote();
    }
    return 0;
}

func::FuncOp AllocationAnnotationPass::propogateRemotableFunction(func::FuncOp funcOp, int index, Type type) {
    OpBuilder builder(funcOp);

    auto &typeFuncs = funcOverrideMap[{funcOp.getName().str(), index}];
    if (auto typeIt = typeFuncs.find(hash_value(type)); typeIt != typeFuncs.end())
        return typeIt->second;

    auto newFuncOp = dyn_cast<func::FuncOp>(funcOp->clone());
    newFuncOp.setName(funcOp.getName().str() + "__" + std::to_string(typeFuncs.size()));

    // llvm::errs() << "create remote function: " << newFuncOp.getName() << ", index " << index << ", type" << type <<
    // "\n";
    auto types =
        newFuncOp->hasAttr("operand_types")
            ? llvm::to_vector(llvm::map_range(newFuncOp->getAttrOfType<ArrayAttr>("operand_types").getValue(),
                                              [](Attribute attr) { return llvm::dyn_cast<TypeAttr>(attr).getValue(); }))
            : llvm::to_vector(newFuncOp.getArgumentTypes());
    types[index] = type;

    newFuncOp->setAttr("operand_types", builder.getTypeArrayAttr(types));
    newFuncOp->setAttr("remote_target", builder.getI64IntegerAttr(1));

    auto id = walkRemoteType(type);

    if (newFuncOp.getNumResults() != 0) {
        auto _retType = newFuncOp.getResultTypes()[0];
        if (isa<LLVM::LLVMPointerType>(_retType))
            newFuncOp->setAttr("rel_types", builder.getTypeArrayAttr(cira::RemoteMemRefType::get(_retType, id)));
        // llvm::errs() << "propogate remotable function: " << newFuncOp << "\n\n\n";
    }

    builder.setInsertionPointAfter(funcOp);
    builder.insert(newFuncOp);

    typeFuncs[hash_value(type)] = newFuncOp;
    return newFuncOp;
}

void AllocationAnnotationPass::propogateRemotableOperator(Operation *op, Type remoteType, Operation *parentOp) {
    auto moduleOp = getOperation();
    SymbolTableCollection symbolTable;
    OpBuilder builder(op);

    // llvm::errs() << "With remote type: " << remoteType << "\n";

    if (auto globalOp = dyn_cast<LLVM::GlobalOp>(op)) {
        // llvm::errs() << "propogate global op\n" << globalOp << "\n";
        auto uses = globalOp.getSymbolUses(moduleOp);
        if (uses.has_value()) {
            for (auto &use : uses.value())
                propogateRemotableOperator(use.getUser(), remoteType, op);
        }
    } else if (auto addressOfOp = dyn_cast<LLVM::AddressOfOp>(op)) {
        // llvm::errs() << "propogate addressof op\n" << addressOfOp << "\n";
        if (!isa<LLVM::GlobalOp>(parentOp))
            return;
        addressOfOp->setAttr("remote_target", builder.getI64IntegerAttr(1));
        remoteType = LLVM::LLVMPointerType::get(remoteType.getContext());
        addressOfOp->setAttr("rel_types", builder.getTypeArrayAttr(remoteType));
        for (auto use : addressOfOp->getUsers())
            propogateRemotableOperator(use, remoteType, addressOfOp);
    } else if (auto loadOp = dyn_cast<LLVM::LoadOp>(op)) {
        // llvm::errs() << "propogate load op\n" << loadOp << "\n";
        if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(remoteType))
            for (auto use : loadOp->getUsers())
                // Opaque pointers - skip element type propagation
                continue;
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
        // llvm::errs() << "propogate call op\n" << callOp << "\n";
        auto callee = callOp.getCallee();
        symbolTable.getSymbolTable(moduleOp);
        auto funcOp = symbolTable.lookupNearestSymbolFrom(callOp, builder.getStringAttr(callee));
        // llvm::errs() << "trace to function: " << callee << ", result: " << funcOp << "\n";

        auto operands = callOp.getArgOperands();
        auto index =
            std::distance(operands.begin(), std::find(operands.begin(), operands.end(), parentOp->getResult(0)));
        auto targetCallee = propogateRemotableFunction(dyn_cast<func::FuncOp>(funcOp), index, remoteType);

        callOp->setAttr("remote_target", builder.getI64IntegerAttr(1));
        if (targetCallee->hasAttr("rel_types"))
            callOp->setAttr("rel_types", targetCallee->getAttr("rel_types"));
        if (targetCallee.getName() != dyn_cast<func::FuncOp>(funcOp).getName())
            callOp->setAttr("remote_callee", builder.getStringAttr(targetCallee.getName()));
    }
}

void AllocationAnnotationPass::propogateRemotable() {
    auto op = getOperation();
    OpBuilder builder(op);
    unsigned remote_id = 1;
    op->walk([this, op, &remote_id, &builder](LLVM::GlobalOp globalOp) {
        if (globalOp.getLinkage() != LLVM::Linkage::External)
            return;
        globalOp->setAttr("remote_target", builder.getI64IntegerAttr(1));
        if (!globalOp->hasAttr("rel_types")) {
            configMap[remote_id] = std::make_tuple(256, transformTypeSize(globalOp.getType()),
                                                   static_cast<uint64_t>(1) * 1024 * 1024 * 1024);
            globalOp->setAttr("rel_types", builder.getTypeArrayAttr(
                                               transformStruct(op->getContext(), globalOp.getType(), remote_id++)));
        }

        auto remoteType = llvm::dyn_cast<TypeAttr>(globalOp->getAttrOfType<ArrayAttr>("rel_types")[0]).getValue();
        propogateRemotableOperator(globalOp, remoteType, op);
    });
}

void AllocationAnnotationPass::duplicateFunctions() {
    std::vector<Operation *> functions;
    auto moduleOp = getOperation();
    OpBuilder builder(moduleOp);

    moduleOp->walk([moduleOp, &builder](func::FuncOp funcOp) {
        if (!funcOp->hasAttr("remote_target"))
            return WalkResult::advance();

        std::vector<CallOpInterface> localCalls{}, remoteCalls{};
        auto uses = funcOp.getSymbolUses(moduleOp);
        if (!uses.has_value())
            return WalkResult::advance();

        for (auto use : uses.value())
            if (auto callOp = dyn_cast<CallOpInterface>(use.getUser())) {
                if (callOp->hasAttr("remote_target"))
                    remoteCalls.push_back(callOp);
                else
                    localCalls.push_back(callOp);
            }

        // llvm::errs() << "Function: " << funcOp.getName() << ", local calls: " << localCalls.size() << "\n";

        if (localCalls.empty())
            return WalkResult::advance();

        auto newFuncOp = dyn_cast<func::FuncOp>(funcOp->clone());
        newFuncOp.setName("_remote__" + funcOp.getName().str());
        builder.setInsertionPointAfter(funcOp);
        builder.insert(newFuncOp);

        funcOp->removeAttr("remote_target");
        funcOp->removeAttr("rel_type");

        // llvm::errs() << "Duplicate Function: " << funcOp.getName() << ", to function: " << newFuncOp.getName() <<
        // "\n";

        // Annotate funcop callers
        for (auto &callOp : remoteCalls)
            callOp->setAttr("remote_callee", builder.getStringAttr(newFuncOp.getName()));
        return WalkResult::advance();
    });
}

void AllocationAnnotationPass::normalizeConfig() {
    if (!memorySizeOption.hasValue())
        return;
    size_t memorySize = memorySizeOption.getValue();
    size_t totalMemory = 0;
    for (auto &kv : configMap) {
        auto [id, config] = kv;
        auto [size, linesize, _] = config;
        totalMemory += size * linesize;
    }
    if (totalMemory < memorySize)
        return;
    // llvm::errs() << "totalMem" << totalMemory << ", memorySize " << memorySize << "\n";
    int factor = (1 + (totalMemory - 1)) / (memorySize * cache_factor);
    for (auto &kv : configMap) {
        auto &[id, config] = kv;
        auto &[size, _linesize, _space] = config;
        // llvm::errs() << "Normalize: id = " << id <<  " size " << size << ", factor " << factor << "\n";
        size = size / factor;
    }
}

void AllocationAnnotationPass::runOnOperation() {
    parseProfilingResults();

    int allocationId = 0;
    auto moduleOp = getOperation();
    bool isAnnotation = annotateOption.hasValue();
    float memoryFactor = memoryFactorOption.hasValue() ? annotateOption.getValue() : 1;
    uint64_t memoryLimit = 4096;

    moduleOp->walk([&memoryFactor, &memoryLimit](func::FuncOp op) {
        OpBuilder builder(op);
        auto funcOp = cast<func::FuncOp>(op);
        auto funcType = funcOp.getFunctionType();

        // annotate on non-allocation part
        if (memoryFactor != 1 && funcType.getNumResults() > 0) {
            op->setAttr("remote_target", builder.getI64IntegerAttr(1));

            auto operandTypes = funcType.getInputs();
            std::vector<Type> newOperandTypes;
            std::transform(operandTypes.begin(), operandTypes.end(), std::back_inserter(newOperandTypes),
                           [memoryLimit](const Type &t) -> Type {
                               return objectLimitFilter(1, t, memoryLimit) ? cira::RemoteMemRefType::get(t, 1) : t;
                           });
            op->setAttr("operand_types", builder.getTypeArrayAttr(newOperandTypes));

            auto retType = cira::RemoteMemRefType::get(funcType.getResult(0), 1);
            op->setAttr("rel_types", builder.getTypeArrayAttr(retType));
        }
    });

    moduleOp->walk([](func::CallOp op) {
        OpBuilder builder(op);
        auto callOp = cast<func::CallOp>(op);
        auto func = callOp.getCallableForCallee();
    });

    moduleOp->walk(
        [isAnnotation, memoryFactor, memoryLimit, &allocationId, &allocationMap = allocationMap](mlir::Operation *op) {
            OpBuilder builder(op);
            if (isa<memref::AllocOp>(op)) {
                int curAllocation = allocationId++;
                // annotate if necessary
                if (isAnnotation)
                    op->setAttr("allocation_id", builder.getI32IntegerAttr(curAllocation));

                if (allocationMap.find(curAllocation) != allocationMap.end()) {
                    if (objectLimitFilter(curAllocation, op->getResult(0).getType(), memoryLimit))
                        op->setAttr("remote_target", builder.getI64IntegerAttr(1));
                }
            }
        });

    propogateRemotable();
    duplicateFunctions();

    // mark remotable functions
    moduleOp->walk([](func::FuncOp funcOp) {
        if (!funcOp->hasAttr("remote_target")) {
            auto result = funcOp->walk([](Operation *op) {
                if (op->hasAttr("remote_target"))
                    return WalkResult::interrupt();
                return WalkResult::advance();
            });
            OpBuilder builder(funcOp);
            if (result.wasInterrupted())
                funcOp->setAttr("remote_target", builder.getI64IntegerAttr(1));
        }
    });

    if (configMap.size() != 0 && !configFile.empty()) {
        normalizeConfig();
        std::ofstream file;
        file.open(configFile);
        size_t token_offset = 0, offset = 0;
        for (const auto &[id, config] : configMap) {
            auto [size, linesize, space] = config;
            file << id << " " << 0 << " " << token_offset << " " << offset << " " << offset << " " << size << " "
                 << linesize << " " << id << "\n";
            token_offset += size;
            offset += space;
        }
        file.close();
    }
}

void AllocationAnnotationPass::parseProfilingResults() {
    if (!memoryProfOption.hasValue() && !cpuProfOption.hasValue() && !runtimeProfOption.hasValue()) {
        // there's no input file. maybe just annotating
        return;
    }

    if (memoryProfOption.hasValue()) {
        std::vector<ProfilingResult> docs;
        loadProfilingFromFile(memoryProfOption.getValue(), docs);

        for (auto &profilingResult : docs) {
            allocationMap[profilingResult.value] = profilingResult;
        }
    }

    return;
}

bool objectDependencyFilter(Operation *op) {
    for (auto user : op->getUsers()) {
        if (isa<func::CallOp>(user)) {
            if (!user->hasAttr("remote_target"))
                return false;
        } else if (isa<memref::CastOp>(user) || isa<memref::ReinterpretCastOp>(user)) {
            // Follow the casting Ops
            return objectDependencyFilter(user);
        }
    }
    return true;
}

bool objectLimitFilter(uint64_t number, Type type, uint64_t memoryLimit) {
    if (isa<MemRefType>(type)) {
        auto memrefType = cast<MemRefType>(type);
        auto shape = memrefType.getShape();
        auto bytes = std::reduce(shape.begin(), shape.end(), 1ULL, std::multiplies<uint64_t>());
        if (bytes < memoryLimit)
            return false;
    }

    return number > 0;
}

std::unique_ptr<Pass> mlir::createAllocationAnnotationPass() { return std::make_unique<AllocationAnnotationPass>(); }
