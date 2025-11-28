//
// OffloadProfile.cpp - Profile-guided offload decision implementation
//

#include "Dialect/OffloadProfile.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <cmath>

using namespace mlir;
using namespace mlir::cira;

//===----------------------------------------------------------------------===//
// JSON Profile Loading
//===----------------------------------------------------------------------===//

LogicalResult mlir::cira::loadOffloadProfile(StringRef filename,
                                             OffloadTimingProfile &profile) {
    auto bufferOrErr = llvm::MemoryBuffer::getFile(filename);
    if (!bufferOrErr) {
        llvm::errs() << "Failed to open profile file: " << filename << "\n";
        return failure();
    }

    auto json = llvm::json::parse(bufferOrErr.get()->getBuffer());
    if (!json) {
        llvm::errs() << "Failed to parse JSON profile: "
                     << llvm::toString(json.takeError()) << "\n";
        return failure();
    }

    auto *root = json->getAsObject();
    if (!root) {
        llvm::errs() << "Profile root is not a JSON object\n";
        return failure();
    }

    // Check if profile data is nested under "baseline" (compiler_profile.json format)
    // or at top level (mcf_baseline_profile.json format)
    const llvm::json::Object *profileData = root;
    if (auto *baseline = root->getObject("baseline")) {
        profileData = baseline;
    }

    // Parse profile type and target
    if (auto type = profileData->getString("profile_type"))
        profile.profile_type = type->str();
    if (auto target = profileData->getString("target"))
        profile.target = target->str();

    // Parse function profiles
    if (auto *functions = profileData->getObject("functions")) {
        for (auto &kv : *functions) {
            FunctionOffloadProfile funcProfile;
            funcProfile.name = kv.first.str();

            if (auto *funcObj = kv.second.getAsObject()) {
                if (auto calls = funcObj->getInteger("calls"))
                    funcProfile.calls = *calls;
                if (auto total = funcObj->getInteger("total_ns"))
                    funcProfile.total_ns = *total;
                if (auto avg = funcObj->getNumber("avg_ns"))
                    funcProfile.avg_ns = *avg;
                if (auto elements = funcObj->getInteger("arcs_processed"))
                    funcProfile.elements_processed = *elements;
                else if (auto elements = funcObj->getInteger("arcs_generated"))
                    funcProfile.elements_processed = *elements;
                if (auto throughput = funcObj->getNumber("throughput_arcs_per_sec"))
                    funcProfile.throughput = *throughput;
                if (auto candidate = funcObj->getBoolean("offload_candidate"))
                    funcProfile.offload_candidate = *candidate;
                if (auto parallelism = funcObj->getString("parallelism"))
                    funcProfile.parallelism_type = parallelism->str();
            }

            profile.functions[funcProfile.name] = funcProfile;
        }
    }

    // Parse overall stats
    if (auto *overall = profileData->getObject("overall")) {
        if (auto iters = overall->getInteger("simplex_iterations"))
            profile.total_simplex_iterations = *iters;
        if (auto total = overall->getInteger("total_execution_ns"))
            profile.total_execution_ns = *total;
    }

    // Parse offload hints
    if (auto *hints = profileData->getObject("offload_hints")) {
        if (auto primary = hints->getString("primary_target"))
            profile.primary_offload_target = primary->str();
        if (auto secondary = hints->getString("secondary_target"))
            profile.secondary_offload_target = secondary->str();
        if (auto speedup = hints->getNumber("expected_speedup_pricing"))
            profile.expected_speedup = *speedup;
        if (auto minArcs = hints->getInteger("min_arcs_for_offload"))
            profile.min_elements_for_offload = *minArcs;
        if (auto factor = hints->getNumber("data_transfer_cost_factor"))
            profile.data_transfer_cost_factor = *factor;
    }

    // Parse timing data (if from Vortex profiling)
    if (auto *timing = root->getObject("timing")) {
        if (auto h2d = timing->getInteger("h2d_latency_ns"))
            profile.h2d_latency_ns = *h2d;
        if (auto kernel = timing->getInteger("kernel_latency_ns"))
            profile.kernel_latency_ns = *kernel;
        if (auto d2h = timing->getInteger("d2h_latency_ns"))
            profile.d2h_latency_ns = *d2h;
    }

    if (auto *bandwidth = root->getObject("bandwidth")) {
        if (auto h2d = bandwidth->getNumber("h2d_gbps"))
            profile.h2d_bandwidth_gbps = *h2d;
        if (auto d2h = bandwidth->getNumber("d2h_gbps"))
            profile.d2h_bandwidth_gbps = *d2h;
    }

    if (auto *prefetch = root->getObject("prefetch_hints")) {
        if (auto dist = prefetch->getInteger("optimal_distance_bytes"))
            profile.optimal_prefetch_distance = *dist;
    }

    return success();
}

//===----------------------------------------------------------------------===//
// Offload Decision Logic
//===----------------------------------------------------------------------===//

OffloadStrategy mlir::cira::computeOffloadStrategy(
    const OffloadTimingProfile &profile,
    StringRef functionName,
    uint64_t estimatedElements,
    uint64_t dataTransferBytes) {

    OffloadStrategy strategy;
    strategy.decision = OffloadDecision::CPU_ONLY;
    strategy.min_elements = profile.min_elements_for_offload;
    strategy.expected_speedup = 1.0;
    strategy.target_device = "cpu";

    // Find function profile
    auto it = profile.functions.find(functionName.str());
    if (it == profile.functions.end()) {
        // No profile data for this function
        return strategy;
    }

    const FunctionOffloadProfile &funcProfile = it->second;

    // Check if function is an offload candidate
    if (!funcProfile.offload_candidate) {
        return strategy;
    }

    // Compute expected speedup based on parallelism type
    double baseSpeedup = 1.0;
    if (funcProfile.parallelism_type == "embarrassingly_parallel") {
        baseSpeedup = 10.0;  // High parallelism potential
    } else if (funcProfile.parallelism_type == "reduction") {
        baseSpeedup = 5.0;   // Good for GPU reduction
    } else if (funcProfile.parallelism_type == "data_dependent") {
        baseSpeedup = 3.0;   // Limited by dependencies
    } else {
        baseSpeedup = 1.0;   // Not suitable for GPU
    }

    // Adjust for data transfer cost
    double transferTime = 0.0;
    if (profile.h2d_bandwidth_gbps > 0 && profile.d2h_bandwidth_gbps > 0) {
        double h2dTime = dataTransferBytes / (profile.h2d_bandwidth_gbps * 1e9);
        double d2hTime = dataTransferBytes / (profile.d2h_bandwidth_gbps * 1e9);
        transferTime = h2dTime + d2hTime;
    }

    // Estimate kernel time on GPU
    double cpuTime = funcProfile.avg_ns * (estimatedElements / std::max(1UL, funcProfile.elements_processed / funcProfile.calls));
    double gpuKernelTime = cpuTime / baseSpeedup;
    double totalGpuTime = gpuKernelTime + transferTime * 1e9;  // Convert to ns

    // Compute actual speedup
    double actualSpeedup = cpuTime / totalGpuTime;
    strategy.expected_speedup = actualSpeedup;

    // Make decision
    if (actualSpeedup > 1.5) {
        if (estimatedElements >= profile.min_elements_for_offload) {
            strategy.decision = OffloadDecision::GPU_ALWAYS;
            strategy.target_device = "vortex";
        } else {
            strategy.decision = OffloadDecision::GPU_CONDITIONAL;
            strategy.target_device = "vortex";
        }
    } else if (actualSpeedup > 1.0) {
        // Marginal benefit - conditional offload
        strategy.decision = OffloadDecision::GPU_CONDITIONAL;
        strategy.target_device = "vortex";
    }

    return strategy;
}

bool mlir::cira::shouldOffloadFunction(
    const OffloadTimingProfile &profile,
    StringRef functionName,
    uint64_t estimatedElements) {

    auto it = profile.functions.find(functionName.str());
    if (it == profile.functions.end())
        return false;

    const FunctionOffloadProfile &funcProfile = it->second;
    if (!funcProfile.offload_candidate)
        return false;

    return estimatedElements >= profile.min_elements_for_offload;
}

double mlir::cira::getExpectedSpeedup(
    const OffloadTimingProfile &profile,
    StringRef functionName) {

    auto it = profile.functions.find(functionName.str());
    if (it == profile.functions.end())
        return 1.0;

    const FunctionOffloadProfile &funcProfile = it->second;

    if (funcProfile.parallelism_type == "embarrassingly_parallel")
        return 10.0;
    else if (funcProfile.parallelism_type == "reduction")
        return 5.0;
    else if (funcProfile.parallelism_type == "data_dependent")
        return 3.0;
    else
        return 1.0;
}

//===----------------------------------------------------------------------===//
// Profile-Guided Offload Pass Implementation
//===----------------------------------------------------------------------===//

void ProfileGuidedOffloadPass::runOnOperation() {
    auto module = getOperation();

    // Load profile if specified
    if (profilePathOption.hasValue()) {
        if (failed(loadOffloadProfile(profilePathOption.getValue(), profile))) {
            llvm::errs() << "Warning: Could not load offload profile, using defaults\n";
        } else {
            llvm::errs() << "Loaded offload profile from: " << profilePathOption.getValue() << "\n";
            llvm::errs() << "  Primary target: " << profile.primary_offload_target << "\n";
            llvm::errs() << "  Expected speedup: " << profile.expected_speedup << "x\n";
        }
    }

    OpBuilder builder(module.getContext());

    // Process each function
    module.walk([&](func::FuncOp funcOp) {
        StringRef funcName = funcOp.getName();

        // Estimate data transfer and element count
        uint64_t estimatedElements = 0;
        uint64_t dataTransferBytes = estimateDataTransfer(funcOp);

        // Check for loops to estimate trip count
        funcOp.walk([&](Operation *op) {
            if (isa<scf::ForOp, scf::WhileOp, scf::ParallelOp>(op)) {
                estimatedElements += estimateLoopTripCount(op);
            }
        });

        // Use minimum from options if no estimation
        if (estimatedElements == 0)
            estimatedElements = minElementsOption.getValue();

        // Compute offload strategy
        OffloadStrategy strategy;
        if (forceOffloadOption.getValue()) {
            strategy.decision = OffloadDecision::GPU_ALWAYS;
            strategy.target_device = "vortex";
            strategy.expected_speedup = profile.expected_speedup;
            strategy.min_elements = minElementsOption.getValue();
        } else {
            strategy = computeOffloadStrategy(profile, funcName, estimatedElements, dataTransferBytes);
        }

        // Check speedup threshold
        if (strategy.expected_speedup < speedupThresholdOption.getValue() &&
            !forceOffloadOption.getValue()) {
            strategy.decision = OffloadDecision::CPU_ONLY;
        }

        // Annotate function
        annotateFunction(funcOp, strategy);

        // Annotate loops within function
        funcOp.walk([&](Operation *op) {
            if (isa<scf::ForOp, scf::ParallelOp>(op)) {
                annotateLoop(op, strategy);
            }
        });
    });
}

void ProfileGuidedOffloadPass::annotateFunction(func::FuncOp funcOp,
                                                const OffloadStrategy &strategy) {
    OpBuilder builder(funcOp);

    switch (strategy.decision) {
    case OffloadDecision::GPU_ALWAYS:
        funcOp->setAttr("offload_target",
                        builder.getStringAttr(strategy.target_device));
        funcOp->setAttr("offload_decision",
                        builder.getStringAttr("always"));
        funcOp->setAttr("expected_speedup",
                        builder.getF64FloatAttr(strategy.expected_speedup));
        funcOp->setAttr("min_elements",
                        builder.getI64IntegerAttr(strategy.min_elements));
        break;

    case OffloadDecision::GPU_CONDITIONAL:
        funcOp->setAttr("offload_target",
                        builder.getStringAttr(strategy.target_device));
        funcOp->setAttr("offload_decision",
                        builder.getStringAttr("conditional"));
        funcOp->setAttr("expected_speedup",
                        builder.getF64FloatAttr(strategy.expected_speedup));
        funcOp->setAttr("min_elements",
                        builder.getI64IntegerAttr(strategy.min_elements));
        break;

    case OffloadDecision::HYBRID:
        funcOp->setAttr("offload_target",
                        builder.getStringAttr("hetero"));
        funcOp->setAttr("offload_decision",
                        builder.getStringAttr("hybrid"));
        break;

    case OffloadDecision::CPU_ONLY:
    default:
        // No annotation needed for CPU-only
        break;
    }
}

void ProfileGuidedOffloadPass::annotateLoop(Operation *loopOp,
                                           const OffloadStrategy &strategy) {
    if (strategy.decision == OffloadDecision::CPU_ONLY)
        return;

    OpBuilder builder(loopOp);

    // Mark loop for GPU execution
    loopOp->setAttr("gpu_offload",
                    builder.getStringAttr(strategy.target_device));

    // Add prefetch hints if available
    if (profile.optimal_prefetch_distance > 0) {
        loopOp->setAttr("prefetch_distance",
                        builder.getI64IntegerAttr(profile.optimal_prefetch_distance));
    }

    // Parallelism hints
    if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
        loopOp->setAttr("parallel_strategy",
                        builder.getStringAttr("thread_strided"));
    } else if (auto parallelOp = dyn_cast<scf::ParallelOp>(loopOp)) {
        loopOp->setAttr("parallel_strategy",
                        builder.getStringAttr("block_parallel"));
    }
}

uint64_t ProfileGuidedOffloadPass::estimateLoopTripCount(Operation *loopOp) {
    if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
        // Try to extract constant bounds
        auto lowerBound = forOp.getLowerBound();
        auto upperBound = forOp.getUpperBound();
        auto step = forOp.getStep();

        // Check if bounds are constants
        if (auto lbConst = lowerBound.getDefiningOp<arith::ConstantOp>()) {
            if (auto ubConst = upperBound.getDefiningOp<arith::ConstantOp>()) {
                if (auto stepConst = step.getDefiningOp<arith::ConstantOp>()) {
                    auto lb = cast<IntegerAttr>(lbConst.getValue()).getInt();
                    auto ub = cast<IntegerAttr>(ubConst.getValue()).getInt();
                    auto s = cast<IntegerAttr>(stepConst.getValue()).getInt();
                    if (s > 0)
                        return (ub - lb + s - 1) / s;
                }
            }
        }

        // Default estimate for unknown bounds
        return 10000;
    }

    if (auto parallelOp = dyn_cast<scf::ParallelOp>(loopOp)) {
        // Estimate based on parallel dimensions
        uint64_t total = 1;
        for (auto [lb, ub, step] : llvm::zip(parallelOp.getLowerBound(),
                                             parallelOp.getUpperBound(),
                                             parallelOp.getStep())) {
            if (auto lbConst = lb.getDefiningOp<arith::ConstantOp>()) {
                if (auto ubConst = ub.getDefiningOp<arith::ConstantOp>()) {
                    if (auto stepConst = step.getDefiningOp<arith::ConstantOp>()) {
                        auto lbVal = cast<IntegerAttr>(lbConst.getValue()).getInt();
                        auto ubVal = cast<IntegerAttr>(ubConst.getValue()).getInt();
                        auto sVal = cast<IntegerAttr>(stepConst.getValue()).getInt();
                        if (sVal > 0)
                            total *= (ubVal - lbVal + sVal - 1) / sVal;
                    }
                }
            }
        }
        return total > 1 ? total : 10000;
    }

    return 10000;  // Default estimate
}

uint64_t ProfileGuidedOffloadPass::estimateDataTransfer(func::FuncOp funcOp) {
    uint64_t bytes = 0;

    // Estimate based on argument types
    for (auto argType : funcOp.getArgumentTypes()) {
        if (auto memrefType = dyn_cast<MemRefType>(argType)) {
            // Calculate size from shape
            int64_t elements = 1;
            for (auto dim : memrefType.getShape()) {
                if (dim > 0)
                    elements *= dim;
                else
                    elements *= 1024;  // Default for dynamic dims
            }
            bytes += elements * (memrefType.getElementTypeBitWidth() / 8);
        } else if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(argType)) {
            // Assume 4KB for pointer arguments
            bytes += 4096;
        }
    }

    return bytes;
}

//===----------------------------------------------------------------------===//
// Offload Cost Model Implementation
//===----------------------------------------------------------------------===//

OffloadCostModel::OffloadCostModel(const OffloadTimingProfile &profile)
    : profile_(profile) {}

double OffloadCostModel::estimateCPUTime(StringRef funcName,
                                         uint64_t elements) const {
    auto it = profile_.functions.find(funcName.str());
    if (it == profile_.functions.end())
        return elements * 100.0;  // Default: 100ns per element

    const FunctionOffloadProfile &func = it->second;
    if (func.throughput > 0)
        return elements / func.throughput * 1e9;  // Convert to ns

    return func.avg_ns * elements;
}

double OffloadCostModel::estimateGPUTime(StringRef funcName, uint64_t elements,
                                        uint64_t h2dBytes, uint64_t d2hBytes) const {
    // Kernel execution time
    double kernelTime = estimateCPUTime(funcName, elements) / kDefaultGPUSpeedup;

    // Transfer times
    double h2dBw = profile_.h2d_bandwidth_gbps > 0 ? profile_.h2d_bandwidth_gbps : kDefaultH2DBandwidth;
    double d2hBw = profile_.d2h_bandwidth_gbps > 0 ? profile_.d2h_bandwidth_gbps : kDefaultD2HBandwidth;

    double h2dTime = h2dBytes / (h2dBw * 1e9) * 1e9;  // ns
    double d2hTime = d2hBytes / (d2hBw * 1e9) * 1e9;  // ns

    return kernelTime + h2dTime + d2hTime + kDefaultKernelOverhead;
}

OffloadDecision OffloadCostModel::getRecommendation(StringRef funcName,
                                                   uint64_t elements,
                                                   uint64_t h2dBytes,
                                                   uint64_t d2hBytes) const {
    double cpuTime = estimateCPUTime(funcName, elements);
    double gpuTime = estimateGPUTime(funcName, elements, h2dBytes, d2hBytes);

    double speedup = cpuTime / gpuTime;

    if (speedup > 2.0)
        return OffloadDecision::GPU_ALWAYS;
    else if (speedup > 1.2)
        return OffloadDecision::GPU_CONDITIONAL;
    else
        return OffloadDecision::CPU_ONLY;
}

uint64_t OffloadCostModel::getCrossoverPoint(StringRef funcName,
                                            uint64_t bytesPerElement) const {
    // Find the point where GPU time equals CPU time
    // CPU: elements * cpu_per_element
    // GPU: elements * gpu_per_element + transfer_overhead + kernel_overhead

    auto it = profile_.functions.find(funcName.str());
    double cpuPerElement = 100.0;  // ns
    if (it != profile_.functions.end() && it->second.throughput > 0) {
        cpuPerElement = 1e9 / it->second.throughput;
    }

    double gpuPerElement = cpuPerElement / kDefaultGPUSpeedup;

    double h2dBw = profile_.h2d_bandwidth_gbps > 0 ? profile_.h2d_bandwidth_gbps : kDefaultH2DBandwidth;
    double transferPerElement = bytesPerElement / (h2dBw * 1e9) * 1e9 * 2;  // H2D + D2H

    // Solve: elements * cpu = elements * (gpu + transfer) + overhead
    // elements * (cpu - gpu - transfer) = overhead
    double diff = cpuPerElement - gpuPerElement - transferPerElement;
    if (diff <= 0)
        return UINT64_MAX;  // GPU never faster

    return static_cast<uint64_t>(kDefaultKernelOverhead / diff);
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::cira::createProfileGuidedOffloadPass() {
    return std::make_unique<ProfileGuidedOffloadPass>();
}
