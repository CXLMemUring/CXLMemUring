#include "Conversion/CiraToLLVM.h"
#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  RISCV_VORTEX,  // Vortex RISC-V SIMT (generates NVVM IR)
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
    else if (targetArch == "vortex" || targetArch == "riscv-vortex")
      return TargetArchitecture::RISCV_VORTEX;
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
    else if (triple.isRISCV()) {
      // Check for Vortex-specific markers
      if (triple.getVendorName() == "vortex" ||
          module->hasAttr("vortex.target"))
        return TargetArchitecture::RISCV_VORTEX;
    }
  }

  // Default to heterogeneous for CXLMemUring
  return TargetArchitecture::Heterogeneous;
}

//===----------------------------------------------------------------------===//
// NVVM Intrinsic Generation for Vortex
//===----------------------------------------------------------------------===//

// Generate NVVM thread indexing intrinsics (map to Vortex vx_thread_id via CuPBoP)
static Value generateNVVMThreadIdx(ConversionPatternRewriter &rewriter, Location loc) {
  auto context = rewriter.getContext();
  auto i32Type = IntegerType::get(context, 32);
  auto tidFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.read.ptx.sreg.tid.x");
  return rewriter.create<LLVM::CallOp>(loc, i32Type, tidFunc).getResult();
}

static Value generateNVVMBlockIdx(ConversionPatternRewriter &rewriter, Location loc) {
  auto context = rewriter.getContext();
  auto i32Type = IntegerType::get(context, 32);
  auto bidFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.read.ptx.sreg.ctaid.x");
  return rewriter.create<LLVM::CallOp>(loc, i32Type, bidFunc).getResult();
}

static Value generateNVVMBlockDim(ConversionPatternRewriter &rewriter, Location loc) {
  auto context = rewriter.getContext();
  auto i32Type = IntegerType::get(context, 32);
  auto bdimFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.read.ptx.sreg.ntid.x");
  return rewriter.create<LLVM::CallOp>(loc, i32Type, bdimFunc).getResult();
}

// Generate global thread ID: blockIdx.x * blockDim.x + threadIdx.x
static Value generateGlobalThreadId(ConversionPatternRewriter &rewriter, Location loc) {
  auto tid = generateNVVMThreadIdx(rewriter, loc);
  auto bid = generateNVVMBlockIdx(rewriter, loc);
  auto bdim = generateNVVMBlockDim(rewriter, loc);

  auto mul = rewriter.create<LLVM::MulOp>(loc, bid, bdim);
  return rewriter.create<LLVM::AddOp>(loc, tid, mul);
}

// Generate NVVM warp-level intrinsics (map to Vortex vx_* intrinsics)
static void generateNVVMBarrier(ConversionPatternRewriter &rewriter, Location loc) {
  auto context = rewriter.getContext();
  auto barrierFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.barrier0");
  rewriter.create<LLVM::CallOp>(loc, TypeRange{}, barrierFunc, ValueRange{});
}

static Value generateNVVMBallot(ConversionPatternRewriter &rewriter, Location loc, Value predicate) {
  auto context = rewriter.getContext();
  auto i32Type = IntegerType::get(context, 32);
  auto ballotFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.vote.ballot.sync");

  // All-active mask for warp (0xFFFFFFFF)
  auto mask = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(-1));

  return rewriter.create<LLVM::CallOp>(loc, i32Type, ballotFunc,
                                        ValueRange{mask, predicate}).getResult();
}

static Value generateNVVMShfl(ConversionPatternRewriter &rewriter, Location loc,
                              Value var, Value lane) {
  auto context = rewriter.getContext();
  auto i32Type = IntegerType::get(context, 32);
  auto shflFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.shfl.sync.idx.i32");

  auto mask = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(-1));
  auto warpSize = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(32));

  return rewriter.create<LLVM::CallOp>(loc, i32Type, shflFunc,
                                        ValueRange{mask, var, lane, warpSize}).getResult();
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

  if (arch == TargetArchitecture::RISCV_VORTEX) {
    // NVVM prefetch intrinsic (maps to Vortex vx_prefetch_l1 via CuPBoP)
    // Use NVVM's prefetch.global intrinsic which CuPBoP translates
    auto prefetchFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.prefetch.global");
    auto i32Type = IntegerType::get(context, 32);

    // level=1 (L1 cache), hint=0 (load)
    auto level = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));
    auto hint = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, prefetchFunc,
                                   ValueRange{addr, level, hint});
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

    if (targetArch == TargetArchitecture::RISCV_VORTEX) {
      // Vortex: Use memory fence to flush cache
      // NVVM doesn't have a direct evict, so use a fence to ensure coherency
      auto fenceFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.membar.gl");
      rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fenceFunc, ValueRange{});
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
// New CIRA Operation Lowerings (Memory, Cache, Sync, Control, Stream)
//===----------------------------------------------------------------------===//

/// Lower cira.alloc_cxl to LLVM malloc with CXL NUMA hint
struct AllocCxlOpLowering : public ConversionPattern {
  AllocCxlOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, AllocCxlOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto allocOp = cast<AllocCxlOp>(op);
    Location loc = op->getLoc();
    auto *ctx = rewriter.getContext();

    // Call numa_alloc_onnode or mmap with CXL memory binding
    // For now, lower to standard malloc - real impl would use numa_alloc
    auto mallocFunc = FlatSymbolRefAttr::get(ctx, "malloc");
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto result = rewriter.create<LLVM::CallOp>(
        loc, ptrType, mallocFunc, operands[0]);

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

/// Lower cira.load_async to non-blocking load + prefetch hint
struct LoadAsyncOpLowering : public ConversionPattern {
  TargetArchitecture targetArch;

  LoadAsyncOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
      : ConversionPattern(converter, LoadAsyncOp::getOperationName(), 1,
                          &converter.getContext()), targetArch(arch) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<LoadAsyncOp>(op);
    Location loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto *converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

    // Get the element type
    Type resultType = converter->convertType(loadOp.getType());
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // Issue prefetch for the address
    emitPrefetchForTarget(rewriter, loc, operands[0], targetArch);

    // Create the load (will complete when data arrives)
    auto loadResult = rewriter.create<LLVM::LoadOp>(loc, resultType, operands[0]);

    // For now, the "future" is just the loaded value
    // Real implementation would use io_uring or similar async mechanism
    rewriter.replaceOp(op, loadResult.getResult());
    return success();
  }
};

/// Lower cira.store_async to non-blocking store
struct StoreAsyncOpLowering : public ConversionPattern {
  StoreAsyncOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, StoreAsyncOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Create non-temporal store for CXL memory
    rewriter.create<LLVM::StoreOp>(loc, operands[0], operands[1],
                                    /*alignment=*/0, /*isVolatile=*/false,
                                    /*isNonTemporal=*/true);

    // Return a "completed" future (null for now)
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
    rewriter.replaceOp(op, nullPtr.getResult());
    return success();
  }
};

/// Lower cira.flush to memory fence
struct FlushOpLowering : public ConversionPattern {
  FlushOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, FlushOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Issue memory fence
    rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::seq_cst);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower cira.prefetch_stream to prefetch intrinsics
struct PrefetchStreamOpLowering : public ConversionPattern {
  TargetArchitecture targetArch;

  PrefetchStreamOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
      : ConversionPattern(converter, PrefetchStreamOp::getOperationName(), 1,
                          &converter.getContext()), targetArch(arch) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto prefetchOp = cast<PrefetchStreamOp>(op);
    Location loc = op->getLoc();

    // Issue multiple prefetch hints for the stream
    // For simplicity, prefetch the first few cache lines
    emitPrefetchForTarget(rewriter, loc, operands[0], targetArch);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower cira.prefetch_indirect to pointer-chasing prefetch
struct PrefetchIndirectOpLowering : public ConversionPattern {
  TargetArchitecture targetArch;

  PrefetchIndirectOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
      : ConversionPattern(converter, PrefetchIndirectOp::getOperationName(), 1,
                          &converter.getContext()), targetArch(arch) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto prefetchOp = cast<PrefetchIndirectOp>(op);
    Location loc = op->getLoc();
    auto *ctx = rewriter.getContext();

    // For Vortex target, generate microkernel for pointer chasing
    if (targetArch == TargetArchitecture::RISCV_VORTEX) {
      // Generate NVVM-style kernel call for Vortex
      auto launchFunc = FlatSymbolRefAttr::get(ctx, "__vortex_prefetch_chain");
      auto i64Type = IntegerType::get(ctx, 64);

      auto offset = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, prefetchOp.getNextPtrOffsetAttr());
      auto depth = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, prefetchOp.getDepthAttr());

      rewriter.create<LLVM::CallOp>(loc, TypeRange{}, launchFunc,
                                     ValueRange{operands[0], offset, depth});
    } else {
      // For x86/ARM, just prefetch the first node
      emitPrefetchForTarget(rewriter, loc, operands[0], targetArch);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower cira.future_await - for now just pass through the value
struct FutureAwaitOpLowering : public ConversionPattern {
  FutureAwaitOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, FutureAwaitOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // Future is already the value (simplified implementation)
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};

/// Lower cira.barrier to memory fence
struct BarrierOpLowering : public ConversionPattern {
  BarrierOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, BarrierOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::seq_cst);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower cira.release to free
struct ReleaseOpLowering : public ConversionPattern {
  ReleaseOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, ReleaseOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto *ctx = rewriter.getContext();

    auto freeFunc = FlatSymbolRefAttr::get(ctx, "free");
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, freeFunc, operands[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower cira.stream_create_indirect - create stream state structure
struct StreamCreateIndirectOpLowering : public ConversionPattern {
  StreamCreateIndirectOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, StreamCreateIndirectOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // For now, the stream is just the starting pointer
    // Real impl would allocate a stream descriptor struct
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};

/// Lower cira.prefetch_chain - generate Vortex microkernel
struct PrefetchChainOpLowering : public ConversionPattern {
  TargetArchitecture targetArch;

  PrefetchChainOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
      : ConversionPattern(converter, PrefetchChainOp::getOperationName(), 1,
                          &converter.getContext()), targetArch(arch) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto chainOp = cast<PrefetchChainOp>(op);
    Location loc = op->getLoc();
    auto *ctx = rewriter.getContext();

    if (targetArch == TargetArchitecture::RISCV_VORTEX) {
      // Generate NVVM kernel call
      auto launchFunc = FlatSymbolRefAttr::get(ctx, "__vortex_prefetch_chain_kernel");
      auto i64Type = IntegerType::get(ctx, 64);
      auto depth = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, chainOp.getDepthAttr());

      rewriter.create<LLVM::CallOp>(loc, TypeRange{}, launchFunc,
                                     ValueRange{operands[0], depth});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower cira.peek_stream - return current element
struct PeekStreamOpLowering : public ConversionPattern {
  PeekStreamOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, PeekStreamOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto peekOp = cast<PeekStreamOp>(op);
    Location loc = op->getLoc();
    auto *converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

    Type resultType = converter->convertType(peekOp.getType());

    // Load from current stream position
    auto loadResult = rewriter.create<LLVM::LoadOp>(loc, resultType, operands[0]);
    rewriter.replaceOp(op, loadResult.getResult());
    return success();
  }
};

/// Lower cira.advance_stream - update stream pointer
struct AdvanceStreamOpLowering : public ConversionPattern {
  AdvanceStreamOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, AdvanceStreamOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // Stream advancement is handled by the transformed loop structure
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower cira.offload_start - emit Vortex kernel launch
struct OffloadStartOpLowering : public ConversionPattern {
  TargetArchitecture targetArch;

  OffloadStartOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
      : ConversionPattern(converter, OffloadStartOp::getOperationName(), 1,
                          &converter.getContext()), targetArch(arch) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto offloadOp = cast<OffloadStartOp>(op);
    Location loc = op->getLoc();

    if (targetArch == TargetArchitecture::RISCV_VORTEX) {
      // Emit kernel launch for Vortex
      // The body operations will be lowered separately
    }

    // Inline the body operations
    Block &body = offloadOp.getBody().front();
    for (auto &bodyOp : llvm::make_early_inc_range(body.getOperations())) {
      rewriter.clone(bodyOp);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CXL Type 2 Operation Lowerings (.cache and .mem protocols)
//===----------------------------------------------------------------------===//

/// Lower cira.type2.cache_load to runtime call cira_type2_cache_load
struct Type2CacheLoadOpLowering : public ConversionPattern {
  TargetArchitecture targetArch;

  Type2CacheLoadOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
      : ConversionPattern(converter, Type2CacheLoadOp::getOperationName(), 1,
                          &converter.getContext()), targetArch(arch) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<Type2CacheLoadOp>(op);
    Location loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto *converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

    Type resultType = converter->convertType(loadOp.getType());
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto i64Type = IntegerType::get(ctx, 64);
    auto i8Type = IntegerType::get(ctx, 8);

    // Allocate stack slot for coherency state output
    auto one = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
        rewriter.getI64IntegerAttr(1));
    auto stateAlloca = rewriter.create<LLVM::AllocaOp>(loc, ptrType, i8Type, one);

    // Call cira_type2_cache_load(controller, host_addr, size, &state)
    auto cacheLoadFunc = FlatSymbolRefAttr::get(ctx, "cira_type2_cache_load");
    auto result = rewriter.create<LLVM::CallOp>(
        loc, ptrType, cacheLoadFunc,
        ValueRange{operands[0], operands[0], operands[1], stateAlloca});

    // For RISCV_VORTEX target, also emit speculative prefetch chain
    if (targetArch == TargetArchitecture::RISCV_VORTEX) {
      auto prefetchFunc = FlatSymbolRefAttr::get(ctx, "__vortex_prefetch_chain");
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
          rewriter.getI64IntegerAttr(0));
      auto depth = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
          rewriter.getI64IntegerAttr(4));
      rewriter.create<LLVM::CallOp>(loc, TypeRange{}, prefetchFunc,
          ValueRange{operands[0], offset, depth});
    }

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

/// Lower cira.type2.cache_store to runtime call cira_type2_cache_store
struct Type2CacheStoreOpLowering : public ConversionPattern {
  Type2CacheStoreOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, Type2CacheStoreOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto i64Type = IntegerType::get(ctx, 64);

    // Store value to a temporary, then pass pointer to runtime
    auto one = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
        rewriter.getI64IntegerAttr(1));
    auto valAlloca = rewriter.create<LLVM::AllocaOp>(
        loc, ptrType, operands[0].getType(), one);
    rewriter.create<LLVM::StoreOp>(loc, operands[0], valAlloca);

    // size = sizeof(element) — use constant matching the type
    auto size = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
        rewriter.getI64IntegerAttr(8)); // default 8 bytes

    // Call cira_type2_cache_store(controller, host_addr, data, size)
    auto cacheStoreFunc = FlatSymbolRefAttr::get(ctx, "cira_type2_cache_store");
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, cacheStoreFunc,
        ValueRange{operands[1], operands[1], valAlloca, size});

    // Return null future
    auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
    rewriter.replaceOp(op, nullPtr.getResult());
    return success();
  }
};

/// Lower cira.type2.mem_load to GEP + LLVM LoadOp with non-temporal hint
struct Type2MemLoadOpLowering : public ConversionPattern {
  Type2MemLoadOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, Type2MemLoadOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto memLoadOp = cast<Type2MemLoadOp>(op);
    Location loc = op->getLoc();
    auto *converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

    Type resultType = converter->convertType(memLoadOp.getType());
    if (!resultType)
      return failure();

    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // GEP: dev_ptr + offset
    auto gep = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, resultType, operands[0], ValueRange{operands[1]});

    // Non-temporal load for CXL.mem (device BAR4 mapped memory)
    auto loadResult = rewriter.create<LLVM::LoadOp>(
        loc, resultType, gep, /*alignment=*/0,
        /*isVolatile=*/false, /*isNonTemporal=*/true);

    rewriter.replaceOp(op, loadResult.getResult());
    return success();
  }
};

/// Lower cira.type2.mem_store to GEP + LLVM StoreOp with non-temporal hint
struct Type2MemStoreOpLowering : public ConversionPattern {
  Type2MemStoreOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, Type2MemStoreOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // GEP: dev_ptr + offset
    auto gep = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, operands[0].getType(), operands[1],
        ValueRange{operands[2]});

    // Non-temporal store for CXL.mem (device BAR4 mapped memory)
    rewriter.create<LLVM::StoreOp>(loc, operands[0], gep,
                                    /*alignment=*/0,
                                    /*isVolatile=*/false,
                                    /*isNonTemporal=*/true);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower cira.type2.drain to runtime call + memory fence
struct Type2DrainOpLowering : public ConversionPattern {
  Type2DrainOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, Type2DrainOp::getOperationName(), 1,
                          &converter.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto *ctx = rewriter.getContext();

    // Call cira_type2_drain_delay_buffer(controller)
    // Controller is obtained from global state; emit call with null (runtime resolves)
    auto drainFunc = FlatSymbolRefAttr::get(ctx, "cira_type2_drain_delay_buffer");
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto nullCtrl = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, drainFunc,
                                   ValueRange{nullCtrl});

    // Full memory fence to ensure ordering
    rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::seq_cst);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower cira.offload region - emit kernel or inline
struct OffloadRegionOpLowering : public ConversionPattern {
  TargetArchitecture targetArch;

  OffloadRegionOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
      : ConversionPattern(converter, OffloadRegionOp::getOperationName(), 1,
                          &converter.getContext()), targetArch(arch) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto offloadOp = cast<OffloadRegionOp>(op);
    Location loc = op->getLoc();

    // For Vortex, the region becomes a kernel
    // For x86, inline the operations

    // Inline the body operations
    Block &body = offloadOp.getBody().front();
    SmallVector<Value> results;

    for (auto &bodyOp : llvm::make_early_inc_range(body.getOperations())) {
      if (auto yieldOp = dyn_cast<YieldOp>(&bodyOp)) {
        // Collect yield values as results
        for (auto val : yieldOp.getValues()) {
          results.push_back(val);
        }
      } else {
        rewriter.clone(bodyOp);
      }
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper function for populating conversion patterns
//===----------------------------------------------------------------------===//

static void populateCiraToLLVMConversionPatternsImpl(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    TargetArchitecture arch) {
  // Legacy patterns
  patterns.add<LoadEdgeOpLowering, EvictEdgeOpLowering>(converter, arch);
  patterns.add<LoadNodeOpLowering, GetPaddrOpLowering, CallOpLowering>(converter);

  // New CIRA operation patterns
  patterns.add<AllocCxlOpLowering>(converter);
  patterns.add<LoadAsyncOpLowering>(converter, arch);
  patterns.add<StoreAsyncOpLowering>(converter);
  patterns.add<FlushOpLowering>(converter);
  patterns.add<PrefetchStreamOpLowering>(converter, arch);
  patterns.add<PrefetchIndirectOpLowering>(converter, arch);
  patterns.add<FutureAwaitOpLowering>(converter);
  patterns.add<BarrierOpLowering>(converter);
  patterns.add<ReleaseOpLowering>(converter);

  // Stream operation patterns
  patterns.add<StreamCreateIndirectOpLowering>(converter);
  patterns.add<PrefetchChainOpLowering>(converter, arch);
  patterns.add<PeekStreamOpLowering>(converter);
  patterns.add<AdvanceStreamOpLowering>(converter);

  // Control operation patterns
  patterns.add<OffloadStartOpLowering>(converter, arch);
  patterns.add<OffloadRegionOpLowering>(converter, arch);

  // CXL Type 2 operation patterns
  patterns.add<Type2CacheLoadOpLowering>(converter, arch);
  patterns.add<Type2CacheStoreOpLowering>(converter);
  patterns.add<Type2MemLoadOpLowering>(converter);
  patterns.add<Type2MemStoreOpLowering>(converter);
  patterns.add<Type2DrainOpLowering>(converter);

  // Legacy generic offload (renamed to offload_legacy)
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
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

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
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

    populateCiraToLLVMConversionPatternsImpl(converter, patterns, TargetArchitecture::ARM);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

// Specialized pass for Vortex RISC-V SIMT target
struct ConvertCiraToLLVMVortexPass
    : public PassWrapper<ConvertCiraToLLVMVortexPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCiraToLLVMVortexPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  StringRef getArgument() const final { return "convert-cira-to-llvm-vortex"; }
  StringRef getDescription() const final {
    return "Convert Cira dialect to NVVM IR for Vortex RISC-V SIMT target";
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Set target triple for NVVM (will be translated to RISC-V via CuPBoP)
    module->setAttr("llvm.target_triple",
                    StringAttr::get(&getContext(), "nvptx64-nvidia-cuda"));

    // Mark module as Vortex target for backend processing
    module->setAttr("vortex.target", UnitAttr::get(&getContext()));

    // Set data layout for NVVM
    module->setAttr("llvm.data_layout",
                    StringAttr::get(&getContext(),
                      "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                      "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
                      "v64:64:64-v128:128:128-n16:32:64"));

    // Mark functions as CUDA kernels
    module.walk([&](func::FuncOp func) {
      // Functions containing offload operations become kernels
      bool hasOffload = false;
      func.walk([&](Operation *op) {
        if (isa<LoadEdgeOp, LoadNodeOp, EvictEdgeOp>(op)) {
          hasOffload = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      if (hasOffload) {
        // Mark as CUDA kernel
        func->setAttr("nvvm.kernel", UnitAttr::get(&getContext()));

        // Add kernel metadata (max threads per block)
        func->setAttr("nvvm.maxntid",
                     DenseI32ArrayAttr::get(&getContext(), {256, 1, 1}));
      }
    });

    LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());

    target.addIllegalDialect<RemoteMemDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<cir::CIRDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

    populateCiraToLLVMConversionPatternsImpl(converter, patterns, TargetArchitecture::RISCV_VORTEX);

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
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

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
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();

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

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMVortexPass() {
  return std::make_unique<ConvertCiraToLLVMVortexPass>();
}

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMHeteroPass() {
  return std::make_unique<ConvertCiraToLLVMHeteroPass>();
}
