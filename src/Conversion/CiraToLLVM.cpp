#include "Conversion/CiraToLLVM.h"
#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/CommandLine.h"
// Allow marking CIR dialect ops as legal when running CIRA-only lowering
#include <clang/CIR/Dialect/IR/CIRDialect.h>

using namespace mlir;
using namespace mlir::cira;

enum class TargetArchitecture {
  X86,
  ARM,
  Heterogeneous
};

namespace {

// Command-line options for target architecture
static llvm::cl::opt<std::string> targetArch(
    "target-arch",
    llvm::cl::desc("Target architecture for code generation"),
    llvm::cl::init("auto"));

// Helper function to determine target architecture
TargetArchitecture getTargetArchitecture(ModuleOp module) {
  if (targetArch != "auto") {
    if (targetArch == "x86")
      return TargetArchitecture::X86;
    else if (targetArch == "arm")
      return TargetArchitecture::ARM;
    else if (targetArch == "hetero")
      return TargetArchitecture::Heterogeneous;
  }

  // Check module's target triple attribute
  if (auto tripleAttr = module->getAttrOfType<StringAttr>("llvm.target_triple")) {
    llvm::Triple triple(tripleAttr.getValue());
    if (triple.isX86())
      return TargetArchitecture::X86;
    else if (triple.isARM() || triple.isAArch64())
      return TargetArchitecture::ARM;
  }

  // Default to heterogeneous for CXLMemUring
  return TargetArchitecture::Heterogeneous;
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Convert cira.offload.load_edge to LLVM operations
struct LoadEdgeOpLowering : public ConversionPattern {
  TargetArchitecture targetArch;

  LoadEdgeOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
      : ConversionPattern(converter, LoadEdgeOp::getOperationName(), 1,
                          &converter.getContext()), targetArch(arch) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<LoadEdgeOp>(op);
    Location loc = op->getLoc();
    
    // Get the type converter
    auto *converter = const_cast<LLVMTypeConverter *>(static_cast<const LLVMTypeConverter *>(getTypeConverter()));
    
    // Convert result type
    Type resultType = converter->convertType(loadOp.getType());
    if (!resultType)
      return failure();
    
    // Create LLVM operations to implement load_edge
    // This is a simplified version - you would implement the actual logic here
    // For now, we'll create a placeholder that loads from a pointer
    
    // Calculate the address: base_ptr + index * element_size
    auto indexType = converter->getIndexType();
    auto index = operands[1];
    
    // Get element size (placeholder - you'd calculate this based on the edge struct)
    auto elementSize = rewriter.create<LLVM::ConstantOp>(
        loc, indexType, rewriter.getIntegerAttr(indexType, 32)); // Assuming 32-byte edges
    
    // Calculate offset
    auto offset = rewriter.create<LLVM::MulOp>(loc, index, elementSize);
    
    // GEP to get the element address
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto gep = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, resultType, operands[0], ValueRange(offset));
    
    // Load the value
    auto loadedValue = rewriter.create<LLVM::LoadOp>(loc, resultType, gep);
    
    // Handle prefetch based on target architecture
    if (operands.size() > 2 && operands[2]) {
      // Prefetch will be handled by target-specific lowering
      // For now, emit a generic prefetch intrinsic
      auto context = rewriter.getContext();
      auto prefetchFunc = FlatSymbolRefAttr::get(context, "llvm.prefetch.p0");
      auto i32Type = IntegerType::get(context, 32);
      auto rw = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
      auto locality = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(3));
      auto cacheType = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));
      rewriter.create<LLVM::CallOp>(loc, TypeRange{}, prefetchFunc,
                                     ValueRange{gep, rw, locality, cacheType});
    }
    
    rewriter.replaceOp(op, loadedValue.getResult());
    return success();
  }
};

/// Convert cira.offload.load_node to LLVM operations
struct LoadNodeOpLowering : public ConversionPattern {
  LoadNodeOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, LoadNodeOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<LoadNodeOp>(op);
    Location loc = op->getLoc();
    
    auto *converter = const_cast<LLVMTypeConverter *>(static_cast<const LLVMTypeConverter *>(getTypeConverter()));
    Type resultType = converter->convertType(loadOp.getType());
    if (!resultType)
      return failure();
    
    // Extract field based on field_name attribute
    StringRef fieldName = loadOp.getFieldName();
    
    // This is a placeholder implementation
    // In a real implementation, you would:
    // 1. Extract the appropriate field from the edge element
    // 2. Load the node data from that address
    // 3. Handle prefetching if specified
    
    // For now, just extract a field from the struct
    // You would implement proper struct field extraction here
    Value nodePtr;
    if (fieldName == "from") {
      // Extract 'from' field (assuming it's at offset 0)
      nodePtr = rewriter.create<LLVM::ExtractValueOp>(
          loc, operands[0], ArrayRef<int64_t>{0});
    } else if (fieldName == "to") {
      // Extract 'to' field (assuming it's at offset 1)
      nodePtr = rewriter.create<LLVM::ExtractValueOp>(
          loc, operands[0], ArrayRef<int64_t>{1});
    } else {
      return failure();
    }
    
    rewriter.replaceOp(op, nodePtr);
    return success();
  }
};

/// Convert cira.offload.get_paddr to LLVM operations
struct GetPaddrOpLowering : public ConversionPattern {
  GetPaddrOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, GetPaddrOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // For this operation, we assume the node data already contains
    // or can be converted to a physical address
    // In a real implementation, this might involve calling runtime functions
    
    // For now, just pass through the value as a pointer
    rewriter.replaceOp(op, operands[1]);
    return success();
  }
};

// Helper function to emit architecture-specific prefetch
static void emitPrefetchForTarget(ConversionPatternRewriter &rewriter, Location loc,
                          Value addr, TargetArchitecture arch) {
  auto context = rewriter.getContext();

  if (arch == TargetArchitecture::X86 || arch == TargetArchitecture::Heterogeneous) {
    // x86 prefetch intrinsic: llvm.prefetch(ptr, rw, locality, cache_type)
    auto prefetchFunc = FlatSymbolRefAttr::get(context, "llvm.prefetch.p0");
    auto i32Type = IntegerType::get(context, 32);

    // rw=0 (read), locality=3 (high), cache_type=1 (data)
    auto rw = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
    auto locality = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(3));
    auto cacheType = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, prefetchFunc,
                                   ValueRange{addr, rw, locality, cacheType});
  }

  if (arch == TargetArchitecture::ARM) {
    // ARM/AArch64 prefetch intrinsic: llvm.aarch64.prefetch(ptr, rw, target, stream, keep)
    auto prefetchFunc = FlatSymbolRefAttr::get(context, "llvm.aarch64.prefetch.p0");
    auto i32Type = IntegerType::get(context, 32);

    // rw=0 (read), target=0 (L1), stream=3 (all levels), keep=0 (temporal), stream=1
    auto rw = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
    auto target = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(3));
    auto stream = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
    auto keep = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, prefetchFunc,
                                   ValueRange{addr, rw, target, stream, keep});
  }
}

/// Convert cira.offload.evict_edge to LLVM operations
struct EvictEdgeOpLowering : public ConversionPattern {
  TargetArchitecture targetArch;

  EvictEdgeOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
      : ConversionPattern(converter, EvictEdgeOp::getOperationName(), 1,
                          &converter.getContext()), targetArch(arch) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto context = rewriter.getContext();

    // Emit architecture-specific cache eviction
    if (targetArch == TargetArchitecture::X86 || targetArch == TargetArchitecture::Heterogeneous) {
      // x86: Use clflush instruction
      auto clflushFunc = FlatSymbolRefAttr::get(context, "llvm.x86.sse2.clflush");
      rewriter.create<LLVM::CallOp>(loc, TypeRange{}, clflushFunc, operands[0]);
    }

    if (targetArch == TargetArchitecture::ARM) {
      // ARM: Use DC CIVAC (Clean and Invalidate by VA to PoC)
      auto dcFunc = FlatSymbolRefAttr::get(context, "llvm.aarch64.dc.civac");
      rewriter.create<LLVM::CallOp>(loc, TypeRange{}, dcFunc, operands[0]);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert cira.call to LLVM call
struct CallOpLowering : public ConversionPattern {
  CallOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, CallOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto callOp = cast<CallOp>(op);
    auto *converter = const_cast<LLVMTypeConverter *>(static_cast<const LLVMTypeConverter *>(getTypeConverter()));
    
    // Convert result types
    SmallVector<Type> resultTypes;
    if (failed(converter->convertTypes(callOp.getResultTypes(), resultTypes)))
      return failure();
    
    // Create LLVM call
    auto llvmCallOp = rewriter.create<LLVM::CallOp>(
        op->getLoc(), resultTypes, callOp.getCalleeAttr(), operands);
    
    rewriter.replaceOp(op, llvmCallOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper function for populating conversion patterns
//===----------------------------------------------------------------------===//

static void populateCiraToLLVMConversionPatternsImpl(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    TargetArchitecture arch) {
  patterns.add<LoadEdgeOpLowering, EvictEdgeOpLowering>(converter, arch);
  patterns.add<LoadNodeOpLowering, GetPaddrOpLowering, CallOpLowering>(converter);
  // Trivial lowering for generic cira.offload (no results): erase the op.
  struct GenericOffloadOpLowering : public ConversionPattern {
    GenericOffloadOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, OffloadOp::getOperationName(), 1,
                            &converter.getContext()) {}
    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
      rewriter.eraseOp(op);
      return success();
    }
  };
  patterns.add<GenericOffloadOpLowering>(converter);
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

// Specialized pass for x86 target
struct ConvertCiraToLLVMX86Pass
    : public PassWrapper<ConvertCiraToLLVMX86Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCiraToLLVMX86Pass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  StringRef getArgument() const final { return "convert-cira-to-llvm-x86"; }
  StringRef getDescription() const final {
    return "Convert Cira dialect to LLVM dialect for x86 target";
  }

  void runOnOperation() override {
    auto module = getOperation();
    // Set target triple for x86
    module->setAttr("llvm.target_triple",
                    StringAttr::get(&getContext(), "x86_64-unknown-linux-gnu"));

    LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());

    target.addIllegalDialect<RemoteMemDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    // Permit CIR ops to remain when only lowering CIRA pieces.
    target.addLegalDialect<cir::CIRDialect>();

    populateCiraToLLVMConversionPatternsImpl(converter, patterns, TargetArchitecture::X86);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

// Specialized pass for ARM target
struct ConvertCiraToLLVMARMPass
    : public PassWrapper<ConvertCiraToLLVMARMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCiraToLLVMARMPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  StringRef getArgument() const final { return "convert-cira-to-llvm-arm"; }
  StringRef getDescription() const final {
    return "Convert Cira dialect to LLVM dialect for ARM target";
  }

  void runOnOperation() override {
    auto module = getOperation();
    // Set target triple for ARM
    module->setAttr("llvm.target_triple",
                    StringAttr::get(&getContext(), "aarch64-unknown-linux-gnu"));

    LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());

    target.addIllegalDialect<RemoteMemDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<cir::CIRDialect>();

    populateCiraToLLVMConversionPatternsImpl(converter, patterns, TargetArchitecture::ARM);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

// Heterogeneous pass that partitions code for both architectures
struct ConvertCiraToLLVMHeteroPass
    : public PassWrapper<ConvertCiraToLLVMHeteroPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCiraToLLVMHeteroPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  StringRef getArgument() const final { return "convert-cira-to-llvm-hetero"; }
  StringRef getDescription() const final {
    return "Convert Cira dialect to LLVM dialect for heterogeneous execution";
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Mark functions with target attributes
    module.walk([&](func::FuncOp func) {
      StringRef name = func.getName();
      // Functions with "remote_access" go to ARM
      if (name.contains("remote_access")) {
        func->setAttr("target-cpu", StringAttr::get(&getContext(), "cortex-a53"));
        func->setAttr("target-features", StringAttr::get(&getContext(), "+neon"));
      } else {
        // Computation functions go to x86
        func->setAttr("target-cpu", StringAttr::get(&getContext(), "x86-64"));
        func->setAttr("target-features", StringAttr::get(&getContext(), "+sse4.2,+avx2"));
      }
    });

    LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());

    target.addIllegalDialect<RemoteMemDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<cir::CIRDialect>();

    populateCiraToLLVMConversionPatternsImpl(converter, patterns, TargetArchitecture::Heterogeneous);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

// Original pass with auto-detection
struct ConvertCiraToLLVMPass
    : public PassWrapper<ConvertCiraToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCiraToLLVMPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  StringRef getArgument() const final { return "convert-cira-to-llvm"; }
  StringRef getDescription() const final {
    return "Convert Cira dialect to LLVM dialect (auto-detect target)";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto arch = getTargetArchitecture(module);

    LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());

    target.addIllegalDialect<RemoteMemDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<cir::CIRDialect>();

    populateCiraToLLVMConversionPatternsImpl(converter, patterns, arch);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::cira::populateCiraToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Default to heterogeneous if not specified
  populateCiraToLLVMConversionPatternsImpl(converter, patterns,
                                      TargetArchitecture::Heterogeneous);
}

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMPass() {
  return std::make_unique<ConvertCiraToLLVMPass>();
}

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMX86Pass() {
  return std::make_unique<ConvertCiraToLLVMX86Pass>();
}

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMARMPass() {
  return std::make_unique<ConvertCiraToLLVMARMPass>();
}

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMHeteroPass() {
  return std::make_unique<ConvertCiraToLLVMHeteroPass>();
}
