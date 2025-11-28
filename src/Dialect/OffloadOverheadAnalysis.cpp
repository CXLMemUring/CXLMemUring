//
// OffloadOverheadAnalysis.cpp - Enhanced overhead analysis implementation
//

#include "Dialect/OffloadOverheadAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <queue>

using namespace mlir;
using namespace mlir::cira;

//===----------------------------------------------------------------------===//
// OffloadOverheadAnalysis Implementation
//===----------------------------------------------------------------------===//

OffloadOverheadAnalysis::OffloadOverheadAnalysis(
    const OffloadTimingProfile &profile)
    : profile_(profile) {}

void OffloadOverheadAnalysis::analyzeFunction(func::FuncOp funcOp) {
    // Build dominance info for the function
    domInfo_ = std::make_unique<DominanceInfo>(funcOp);
    postDomInfo_ = std::make_unique<PostDominanceInfo>(funcOp);
}

//===----------------------------------------------------------------------===//
// Liveness Analysis
//===----------------------------------------------------------------------===//

OffloadLivenessState OffloadOverheadAnalysis::computeLiveness(
    Operation *regionOp) {
    OffloadLivenessState state;

    // Collect all operations in the region
    SmallVector<Operation *, 64> regionOps;
    regionOp->walk([&](Operation *op) {
        if (op != regionOp)
            regionOps.push_back(op);
    });

    // Collect values defined in the region
    for (Operation *op : regionOps) {
        for (Value result : op->getResults()) {
            state.defined.insert(result);
        }
    }

    // For scf.for loops, add induction variable and iter_args
    if (auto forOp = dyn_cast<scf::ForOp>(regionOp)) {
        state.defined.insert(forOp.getInductionVar());
        for (auto iterArg : forOp.getRegionIterArgs()) {
            state.defined.insert(iterArg);
        }
    }

    // Collect values used in the region
    for (Operation *op : regionOps) {
        for (Value operand : op->getOperands()) {
            state.used.insert(operand);

            // If value is not defined in region, it's live-in (needs H2D)
            if (!state.defined.contains(operand)) {
                state.liveIn.insert(operand);
            }
        }
    }

    // Determine live-out values (need D2H)
    // A value is live-out if it's defined in the region and used outside
    for (Value defined : state.defined) {
        for (Operation *user : defined.getUsers()) {
            // Check if user is outside the region
            Operation *parent = user->getParentOp();
            bool outsideRegion = true;
            while (parent) {
                if (parent == regionOp) {
                    outsideRegion = false;
                    break;
                }
                parent = parent->getParentOp();
            }

            if (outsideRegion) {
                state.liveOut.insert(defined);
                break;
            }
        }
    }

    // For scf.for, the yielded values are live-out
    if (auto forOp = dyn_cast<scf::ForOp>(regionOp)) {
        for (Value result : forOp.getResults()) {
            state.liveOut.insert(result);
        }
    }

    return state;
}

uint64_t OffloadOverheadAnalysis::estimateValueSize(Value value) {
    Type type = value.getType();

    // Handle memref types
    if (auto memrefType = dyn_cast<MemRefType>(type)) {
        int64_t elements = 1;
        for (int64_t dim : memrefType.getShape()) {
            if (dim > 0)
                elements *= dim;
            else
                elements *= 1024;  // Default for dynamic dimensions
        }
        return elements * (memrefType.getElementTypeBitWidth() / 8);
    }

    // Handle pointer types
    if (isa<LLVM::LLVMPointerType>(type)) {
        return 4096;  // Default estimate for pointers
    }

    // Handle tensor types
    if (auto tensorType = dyn_cast<TensorType>(type)) {
        int64_t elements = 1;
        if (tensorType.hasStaticShape()) {
            for (int64_t dim : tensorType.getShape())
                elements *= dim;
        } else {
            elements = 1024;  // Default for dynamic shapes
        }
        return elements * (tensorType.getElementTypeBitWidth() / 8);
    }

    // Scalar types
    if (auto intType = dyn_cast<IntegerType>(type)) {
        return intType.getWidth() / 8;
    }
    if (type.isF32())
        return 4;
    if (type.isF64())
        return 8;

    return 8;  // Default
}

void OffloadOverheadAnalysis::computeTransferValues(
    Operation *regionOp,
    SmallVectorImpl<TransferValue> &h2dValues,
    SmallVectorImpl<TransferValue> &d2hValues) {

    OffloadLivenessState liveness = computeLiveness(regionOp);

    // Build H2D transfer list (live-in values)
    for (Value val : liveness.liveIn) {
        // Skip constants and block arguments from parent regions
        if (val.getDefiningOp() && isa<arith::ConstantOp>(val.getDefiningOp()))
            continue;

        TransferValue tv;
        tv.value = val;
        tv.type = val.getType();
        tv.sizeBytes = estimateValueSize(val);
        tv.isInput = true;
        tv.isOutput = liveness.liveOut.contains(val);
        tv.definingOp = val.getDefiningOp();
        tv.lastUseOp = nullptr;

        h2dValues.push_back(tv);
    }

    // Build D2H transfer list (live-out values)
    for (Value val : liveness.liveOut) {
        // Skip values already in H2D list with isOutput=true
        bool alreadyInH2D = false;
        for (auto &tv : h2dValues) {
            if (tv.value == val && tv.isOutput) {
                alreadyInH2D = true;
                break;
            }
        }

        if (alreadyInH2D)
            continue;

        TransferValue tv;
        tv.value = val;
        tv.type = val.getType();
        tv.sizeBytes = estimateValueSize(val);
        tv.isInput = false;
        tv.isOutput = true;
        tv.definingOp = val.getDefiningOp();
        tv.lastUseOp = nullptr;

        d2hValues.push_back(tv);
    }
}

//===----------------------------------------------------------------------===//
// Overhead Computation with Heuristics
//===----------------------------------------------------------------------===//

OffloadOverheadAnalysis::OverheadResult
OffloadOverheadAnalysis::computeOverhead(Operation *regionOp,
                                         uint64_t estimatedElements) {
    OverheadResult result;
    result.h2dBytes = 0;
    result.d2hBytes = 0;
    result.shouldOffload = false;

    // Compute transfer values
    SmallVector<TransferValue> h2dValues, d2hValues;
    computeTransferValues(regionOp, h2dValues, d2hValues);

    // Sum transfer bytes
    for (auto &tv : h2dValues) {
        result.h2dBytes += tv.sizeBytes;
    }
    for (auto &tv : d2hValues) {
        result.d2hBytes += tv.sizeBytes;
    }

    // Use profile data for timing calculations
    double h2dBandwidth = profile_.h2d_bandwidth_gbps > 0
                              ? profile_.h2d_bandwidth_gbps
                              : 10.0;  // GB/s
    double d2hBandwidth = profile_.d2h_bandwidth_gbps > 0
                              ? profile_.d2h_bandwidth_gbps
                              : 10.0;  // GB/s

    // Calculate transfer latencies (ns)
    // latency = base_latency + bytes / bandwidth
    uint64_t baseH2DLatency = 100000;  // 100 us base latency
    uint64_t baseD2HLatency = 50000;   // 50 us base latency

    result.h2dLatencyNs = baseH2DLatency +
                          (result.h2dBytes * 1e9) / (h2dBandwidth * 1e9);
    result.d2hLatencyNs = baseD2HLatency +
                          (result.d2hBytes * 1e9) / (d2hBandwidth * 1e9);

    // Get kernel cycles from profile
    result.kernelCycles = profile_.kernel_latency_ns;
    if (result.kernelCycles == 0) {
        // Estimate: 100 cycles per element
        result.kernelCycles = estimatedElements * 100;
    }

    // Scale kernel cycles by element count if profile was for different size
    if (profile_.min_elements_for_offload > 0 && estimatedElements > 0) {
        // Linear scaling assumption
        double scale = static_cast<double>(estimatedElements) /
                       profile_.min_elements_for_offload;
        result.kernelCycles = static_cast<uint64_t>(result.kernelCycles * scale);
    }

    // Total overhead (in ns, assuming 1 GHz clock)
    result.totalOverheadNs = result.h2dLatencyNs + result.kernelCycles +
                             result.d2hLatencyNs;

    // Calculate expected speedup
    // CPU time estimate from profile
    double cpuTimeNs = 0;
    if (profile_.total_execution_ns > 0) {
        // Use profiled CPU time
        cpuTimeNs = profile_.total_execution_ns;
    } else {
        // Estimate: 1000ns per element on CPU
        cpuTimeNs = estimatedElements * 1000.0;
    }

    result.expectedSpeedup = cpuTimeNs / result.totalOverheadNs;

    // Decision heuristics
    // Rule 1: Check minimum speedup threshold
    double minSpeedup = 1.2;  // 20% improvement minimum

    // Rule 2: Check if kernel time dominates transfer time
    double kernelRatio = static_cast<double>(result.kernelCycles) /
                         result.totalOverheadNs;

    // Rule 3: Check element count threshold
    uint64_t minElements = profile_.min_elements_for_offload > 0
                               ? profile_.min_elements_for_offload
                               : 1000;

    // Make decision
    if (result.expectedSpeedup < minSpeedup) {
        result.shouldOffload = false;
        result.reason = "Expected speedup " +
                        std::to_string(result.expectedSpeedup) +
                        "x below threshold " + std::to_string(minSpeedup) + "x";
    } else if (estimatedElements < minElements) {
        result.shouldOffload = false;
        result.reason = "Element count " + std::to_string(estimatedElements) +
                        " below minimum " + std::to_string(minElements);
    } else if (kernelRatio < 0.1) {
        // Transfer dominates - not worth offloading
        result.shouldOffload = false;
        result.reason = "Transfer overhead dominates (kernel only " +
                        std::to_string(kernelRatio * 100) + "% of total)";
    } else {
        result.shouldOffload = true;
        result.reason = "Offload beneficial: " +
                        std::to_string(result.expectedSpeedup) + "x speedup";
    }

    return result;
}

//===----------------------------------------------------------------------===//
// Dominator Tree Based Placement Optimization
//===----------------------------------------------------------------------===//

OffloadOverheadAnalysis::TransferPlacement
OffloadOverheadAnalysis::getOptimalPlacement(Operation *regionOp) {
    TransferPlacement placement;
    placement.h2dInsertPoint = regionOp;
    placement.d2hInsertPoint = regionOp;
    placement.canHoistH2D = false;
    placement.canSinkD2H = false;

    if (!domInfo_)
        return placement;

    // Find the parent loop (if any)
    Operation *parentLoop = regionOp->getParentOfType<scf::ForOp>();
    if (!parentLoop)
        parentLoop = regionOp->getParentOfType<scf::WhileOp>();

    if (!parentLoop)
        return placement;  // No loop to optimize

    // Get live-in values
    SmallVector<TransferValue> h2dValues, d2hValues;
    computeTransferValues(regionOp, h2dValues, d2hValues);

    // Check if H2D can be hoisted out of loop
    // Conditions: All live-in values are defined before the loop
    bool canHoist = true;
    for (auto &tv : h2dValues) {
        if (tv.definingOp) {
            if (!domInfo_->dominates(tv.definingOp, parentLoop)) {
                canHoist = false;
                break;
            }
        }
    }

    if (canHoist) {
        placement.h2dInsertPoint = parentLoop;
        placement.canHoistH2D = true;
    }

    // Check if D2H can be sunk out of loop
    // Conditions: All live-out values are only used after the loop
    bool canSink = true;
    for (auto &tv : d2hValues) {
        for (Operation *user : tv.value.getUsers()) {
            // Check if user is inside the loop
            Operation *parent = user;
            while (parent && parent != parentLoop) {
                parent = parent->getParentOp();
            }
            if (parent == parentLoop) {
                canSink = false;
                break;
            }
        }
        if (!canSink)
            break;
    }

    if (canSink) {
        placement.d2hInsertPoint = parentLoop->getNextNode();
        placement.canSinkD2H = true;
    }

    return placement;
}

//===----------------------------------------------------------------------===//
// Enhanced Offload Analysis Pass
//===----------------------------------------------------------------------===//

void EnhancedOffloadAnalysisPass::runOnOperation() {
    auto module = getOperation();

    // Load profile
    if (profilePathOption.hasValue()) {
        if (failed(loadOffloadProfile(profilePathOption.getValue(), profile_))) {
            llvm::errs() << "Warning: Could not load offload profile\n";
        }
    }

    // Load experiment results if provided
    if (experimentResultsOption.hasValue()) {
        if (failed(loadExperimentResults(experimentResultsOption.getValue(),
                                         profile_))) {
            llvm::errs() << "Warning: Could not load experiment results\n";
        }
    }

    OffloadOverheadAnalysis analysis(profile_);

    // Process each function
    module.walk([&](func::FuncOp funcOp) {
        analyzeAndAnnotate(funcOp, analysis);
    });
}

void EnhancedOffloadAnalysisPass::analyzeAndAnnotate(
    func::FuncOp funcOp, OffloadOverheadAnalysis &analysis) {

    analysis.analyzeFunction(funcOp);
    OpBuilder builder(funcOp);

    // Find loops to analyze
    funcOp.walk([&](Operation *op) {
        if (!isa<scf::ForOp, scf::ParallelOp>(op))
            return;

        // Estimate elements
        uint64_t elements = 10000;  // Default
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            auto lb = forOp.getLowerBound();
            auto ub = forOp.getUpperBound();
            if (auto lbConst = lb.getDefiningOp<arith::ConstantOp>()) {
                if (auto ubConst = ub.getDefiningOp<arith::ConstantOp>()) {
                    auto lbVal = cast<IntegerAttr>(lbConst.getValue()).getInt();
                    auto ubVal = cast<IntegerAttr>(ubConst.getValue()).getInt();
                    elements = ubVal - lbVal;
                }
            }
        }

        // Compute overhead
        auto result = analysis.computeOverhead(op, elements);

        // Get optimal placement
        auto placement = analysis.getOptimalPlacement(op);

        // Annotate operation
        op->setAttr("h2d_bytes", builder.getI64IntegerAttr(result.h2dBytes));
        op->setAttr("d2h_bytes", builder.getI64IntegerAttr(result.d2hBytes));
        op->setAttr("h2d_latency_ns",
                    builder.getI64IntegerAttr(result.h2dLatencyNs));
        op->setAttr("d2h_latency_ns",
                    builder.getI64IntegerAttr(result.d2hLatencyNs));
        op->setAttr("kernel_cycles",
                    builder.getI64IntegerAttr(result.kernelCycles));
        op->setAttr("total_overhead_ns",
                    builder.getI64IntegerAttr(result.totalOverheadNs));
        op->setAttr("expected_speedup",
                    builder.getF64FloatAttr(result.expectedSpeedup));
        op->setAttr("should_offload",
                    builder.getBoolAttr(result.shouldOffload));
        op->setAttr("offload_reason", builder.getStringAttr(result.reason));

        // Placement hints
        op->setAttr("can_hoist_h2d", builder.getBoolAttr(placement.canHoistH2D));
        op->setAttr("can_sink_d2h", builder.getBoolAttr(placement.canSinkD2H));

        // Print verbose output
        if (verboseOption.getValue()) {
            llvm::errs() << "\n=== Offload Analysis for loop at "
                         << op->getLoc() << " ===\n";
            llvm::errs() << "  H2D bytes: " << result.h2dBytes << "\n";
            llvm::errs() << "  D2H bytes: " << result.d2hBytes << "\n";
            llvm::errs() << "  H2D latency: " << result.h2dLatencyNs << " ns\n";
            llvm::errs() << "  D2H latency: " << result.d2hLatencyNs << " ns\n";
            llvm::errs() << "  Kernel cycles: " << result.kernelCycles << "\n";
            llvm::errs() << "  Total overhead: " << result.totalOverheadNs
                         << " ns\n";
            llvm::errs() << "  Expected speedup: " << result.expectedSpeedup
                         << "x\n";
            llvm::errs() << "  Should offload: "
                         << (result.shouldOffload ? "YES" : "NO") << "\n";
            llvm::errs() << "  Reason: " << result.reason << "\n";
            if (placement.canHoistH2D)
                llvm::errs() << "  Can hoist H2D out of loop\n";
            if (placement.canSinkD2H)
                llvm::errs() << "  Can sink D2H out of loop\n";
        }
    });
}

//===----------------------------------------------------------------------===//
// Profile Export and Loading
//===----------------------------------------------------------------------===//

LogicalResult mlir::cira::loadExperimentResults(
    StringRef experimentResultsPath, OffloadTimingProfile &profile) {

    auto bufferOrErr = llvm::MemoryBuffer::getFile(experimentResultsPath);
    if (!bufferOrErr) {
        llvm::errs() << "Failed to open experiment results: "
                     << experimentResultsPath << "\n";
        return failure();
    }

    auto json = llvm::json::parse(bufferOrErr.get()->getBuffer());
    if (!json) {
        llvm::errs() << "Failed to parse experiment results JSON\n";
        return failure();
    }

    auto *root = json->getAsObject();
    if (!root)
        return failure();

    // Parse x86 baseline
    if (auto *x86 = root->getObject("x86_baseline")) {
        if (auto total = x86->getInteger("total_execution_ns"))
            profile.total_execution_ns = *total;
    }

    // Parse Vortex execution
    if (auto *vortex = root->getObject("vortex_execution")) {
        if (auto cycles = vortex->getInteger("kernel_cycles"))
            profile.kernel_latency_ns = *cycles;
        if (auto h2d = vortex->getInteger("h2d_latency_ns"))
            profile.h2d_latency_ns = *h2d;
        if (auto d2h = vortex->getInteger("d2h_latency_ns"))
            profile.d2h_latency_ns = *d2h;
    }

    // Parse analysis
    if (auto *analysis = root->getObject("analysis")) {
        if (auto speedup = analysis->getNumber("kernel_speedup"))
            profile.expected_speedup = *speedup;
    }

    // Parse offload hints
    if (auto *hints = root->getObject("offload_hints")) {
        if (auto target = hints->getString("primary_target"))
            profile.primary_offload_target = target->str();
        if (auto minArcs = hints->getInteger("min_arcs_for_offload"))
            profile.min_elements_for_offload = *minArcs;
    }

    // Set default bandwidth values
    profile.h2d_bandwidth_gbps = 10.0;
    profile.d2h_bandwidth_gbps = 10.0;

    return success();
}

LogicalResult mlir::cira::exportProfileForCompiler(
    StringRef experimentResultsPath, StringRef outputPath,
    const OffloadOverheadAnalysis::OverheadResult &analysisResult) {

    // Create JSON output
    llvm::json::Object output;

    // Profile metadata
    output["profile_type"] = "compiler_offload_profile";
    output["version"] = "2.0";

    // Overhead analysis results
    llvm::json::Object overhead;
    overhead["h2d_bytes"] = static_cast<int64_t>(analysisResult.h2dBytes);
    overhead["d2h_bytes"] = static_cast<int64_t>(analysisResult.d2hBytes);
    overhead["h2d_latency_ns"] =
        static_cast<int64_t>(analysisResult.h2dLatencyNs);
    overhead["d2h_latency_ns"] =
        static_cast<int64_t>(analysisResult.d2hLatencyNs);
    overhead["kernel_cycles"] =
        static_cast<int64_t>(analysisResult.kernelCycles);
    overhead["total_overhead_ns"] =
        static_cast<int64_t>(analysisResult.totalOverheadNs);
    output["overhead_analysis"] = std::move(overhead);

    // Decision
    llvm::json::Object decision;
    decision["expected_speedup"] = analysisResult.expectedSpeedup;
    decision["should_offload"] = analysisResult.shouldOffload;
    decision["reason"] = analysisResult.reason;
    output["offload_decision"] = std::move(decision);

    // Write to file
    std::error_code ec;
    llvm::raw_fd_ostream os(outputPath, ec);
    if (ec) {
        llvm::errs() << "Failed to open output file: " << outputPath << "\n";
        return failure();
    }

    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(output)));
    return success();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::cira::createEnhancedOffloadAnalysisPass() {
    return std::make_unique<EnhancedOffloadAnalysisPass>();
}
