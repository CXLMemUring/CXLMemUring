//===- TOSAOffloadAnalysis.cpp - TOSA Offload Analysis --------------===//
//
// This file implements analysis for identifying TOSA operations that would
// benefit from remote memory offloading and cost modeling.
//
//===----------------------------------------------------------------------===//

#include "Analysis/TOSAOffloadAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Visitors.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tosa-offload-analysis"

using namespace mlir;
using namespace mlir::cira;

//===----------------------------------------------------------------------===//
// TOSAGraphAnalysis Implementation
//===----------------------------------------------------------------------===//

void TOSAGraphAnalysis::analyzeFunction(func::FuncOp func) {
  opProfiles_.clear();
  dataFlowGraph_.clear();
  criticalPath_.clear();

  // Build operation profiles
  func.walk([&](Operation *op) {
    if (auto matmulOp = dyn_cast<tosa::MatMulOp>(op)) {
      opProfiles_.push_back(profileMatMul(matmulOp));
    } else if (auto convOp = dyn_cast<tosa::Conv2DOp>(op)) {
      opProfiles_.push_back(profileConv2D(convOp));
    } else if (auto reduceOp = dyn_cast<tosa::ReduceSumOp>(op)) {
      opProfiles_.push_back(profileReduce(reduceOp));
    } else if (isa<tosa::AddOp, tosa::MulOp, tosa::SubOp>(op)) {
      opProfiles_.push_back(profileElementwise(op));
    }
  });

  buildDataFlowGraph(func);
  computeCriticalPath();
}

TensorOpProfile TOSAGraphAnalysis::profileMatMul(tosa::MatMulOp op) {
  TensorOpProfile profile;
  profile.op = op.getOperation();

  auto lhsType = op.getA().getType().cast<TensorType>();
  auto rhsType = op.getB().getType().cast<TensorType>();
  auto resultType = op.getResult().getType().cast<TensorType>();

  if (lhsType.hasStaticShape() && rhsType.hasStaticShape()) {
    int64_t m = lhsType.getShape()[0];
    int64_t k = lhsType.getShape()[1];
    int64_t n = rhsType.getShape()[1];

    // Calculate FLOPs for matrix multiplication: 2*M*K*N
    profile.flopCount = 2 * m * k * n;

    // Calculate memory footprint
    int64_t elemSize = lhsType.getElementTypeBitWidth() / 8;
    int64_t lhsSize = lhsType.getNumElements() * elemSize;
    int64_t rhsSize = rhsType.getNumElements() * elemSize;
    int64_t resultSize = resultType.getNumElements() * elemSize;
    profile.memoryFootprint = lhsSize + rhsSize + resultSize;

    // Analyze access patterns
    AccessInfo lhsAccess;
    lhsAccess.pattern = MemoryAccessPattern::SEQUENTIAL;
    lhsAccess.accessCount = m * k * n; // Each element accessed n times
    lhsAccess.dataSize = lhsSize;
    lhsAccess.intensity = static_cast<double>(profile.flopCount) / lhsSize;
    lhsAccess.isReusable = true;
    profile.inputAccess.push_back(lhsAccess);

    AccessInfo rhsAccess;
    rhsAccess.pattern = MemoryAccessPattern::STRIDED;
    rhsAccess.accessCount = m * k * n; // Each element accessed m times
    rhsAccess.dataSize = rhsSize;
    rhsAccess.intensity = static_cast<double>(profile.flopCount) / rhsSize;
    rhsAccess.isReusable = true;
    profile.inputAccess.push_back(rhsAccess);

    AccessInfo resultAccess;
    resultAccess.pattern = MemoryAccessPattern::SEQUENTIAL;
    resultAccess.accessCount = m * n;
    resultAccess.dataSize = resultSize;
    resultAccess.intensity = static_cast<double>(profile.flopCount) / resultSize;
    resultAccess.isReusable = false;
    profile.outputAccess.push_back(resultAccess);

    // Compute intensity (FLOPs per byte)
    profile.computeIntensity = static_cast<double>(profile.flopCount) / profile.memoryFootprint;
    profile.hasDataReuse = true; // Matrix elements are reused
  }

  return profile;
}

TensorOpProfile TOSAGraphAnalysis::profileConv2D(tosa::Conv2DOp op) {
  TensorOpProfile profile;
  profile.op = op.getOperation();

  auto inputType = op.getInput().getType().cast<TensorType>();
  auto weightType = op.getWeight().getType().cast<TensorType>();
  auto resultType = op.getResult().getType().cast<TensorType>();

  if (inputType.hasStaticShape() && weightType.hasStaticShape()) {
    // NHWC format
    auto inputShape = inputType.getShape();
    auto weightShape = weightType.getShape();
    auto resultShape = resultType.getShape();

    int64_t batch = inputShape[0];
    int64_t inputH = inputShape[1];
    int64_t inputW = inputShape[2];
    int64_t inputC = inputShape[3];
    int64_t outputC = weightShape[0];
    int64_t kernelH = weightShape[1];
    int64_t kernelW = weightShape[2];
    int64_t outputH = resultShape[1];
    int64_t outputW = resultShape[2];

    // Calculate FLOPs for convolution
    profile.flopCount = batch * outputH * outputW * outputC * inputC * kernelH * kernelW * 2;

    // Calculate memory footprint
    int64_t elemSize = inputType.getElementTypeBitWidth() / 8;
    int64_t inputSize = inputType.getNumElements() * elemSize;
    int64_t weightSize = weightType.getNumElements() * elemSize;
    int64_t resultSize = resultType.getNumElements() * elemSize;
    profile.memoryFootprint = inputSize + weightSize + resultSize;

    // Analyze access patterns for convolution
    AccessInfo inputAccess;
    inputAccess.pattern = MemoryAccessPattern::BLOCK_SPARSE; // Sliding window access
    inputAccess.accessCount = batch * outputH * outputW * inputC * kernelH * kernelW;
    inputAccess.dataSize = inputSize;
    inputAccess.intensity = static_cast<double>(profile.flopCount) / inputSize;
    inputAccess.isReusable = true; // Input windows overlap
    profile.inputAccess.push_back(inputAccess);

    AccessInfo weightAccess;
    weightAccess.pattern = MemoryAccessPattern::SEQUENTIAL;
    weightAccess.accessCount = batch * outputH * outputW * outputC * kernelH * kernelW;
    weightAccess.dataSize = weightSize;
    weightAccess.intensity = static_cast<double>(profile.flopCount) / weightSize;
    weightAccess.isReusable = true; // Weights reused across spatial locations
    profile.inputAccess.push_back(weightAccess);

    AccessInfo resultAccess;
    resultAccess.pattern = MemoryAccessPattern::SEQUENTIAL;
    resultAccess.accessCount = batch * outputH * outputW * outputC;
    resultAccess.dataSize = resultSize;
    resultAccess.intensity = static_cast<double>(profile.flopCount) / resultSize;
    resultAccess.isReusable = false;
    profile.outputAccess.push_back(resultAccess);

    profile.computeIntensity = static_cast<double>(profile.flopCount) / profile.memoryFootprint;
    profile.hasDataReuse = true; // High reuse in convolution
  }

  return profile;
}

TensorOpProfile TOSAGraphAnalysis::profileReduce(tosa::ReduceSumOp op) {
  TensorOpProfile profile;
  profile.op = op.getOperation();

  auto inputType = op.getInput().getType().cast<TensorType>();
  auto resultType = op.getResult().getType().cast<TensorType>();

  if (inputType.hasStaticShape()) {
    int64_t inputElements = inputType.getNumElements();
    int64_t resultElements = resultType.getNumElements();

    // FLOPs for reduction (one add per input element)
    profile.flopCount = inputElements;

    // Memory footprint
    int64_t elemSize = inputType.getElementTypeBitWidth() / 8;
    int64_t inputSize = inputElements * elemSize;
    int64_t resultSize = resultElements * elemSize;
    profile.memoryFootprint = inputSize + resultSize;

    // Analyze access patterns
    AccessInfo inputAccess;
    inputAccess.pattern = MemoryAccessPattern::SEQUENTIAL;
    inputAccess.accessCount = inputElements;
    inputAccess.dataSize = inputSize;
    inputAccess.intensity = static_cast<double>(profile.flopCount) / inputSize;
    inputAccess.isReusable = false; // Each element read once
    profile.inputAccess.push_back(inputAccess);

    AccessInfo resultAccess;
    resultAccess.pattern = MemoryAccessPattern::SEQUENTIAL;
    resultAccess.accessCount = resultElements;
    resultAccess.dataSize = resultSize;
    resultAccess.intensity = static_cast<double>(profile.flopCount) / resultSize;
    resultAccess.isReusable = false;
    profile.outputAccess.push_back(resultAccess);

    profile.computeIntensity = static_cast<double>(profile.flopCount) / profile.memoryFootprint;
    profile.hasDataReuse = false; // Limited reuse in reductions
  }

  return profile;
}

TensorOpProfile TOSAGraphAnalysis::profileElementwise(Operation *op) {
  TensorOpProfile profile;
  profile.op = op;

  // Get input and output tensors
  auto inputs = op->getOperands();
  auto results = op->getResults();

  if (!inputs.empty() && !results.empty()) {
    auto inputType = inputs[0].getType().cast<TensorType>();
    auto resultType = results[0].getType().cast<TensorType>();

    if (inputType.hasStaticShape()) {
      int64_t numElements = inputType.getNumElements();

      // FLOPs for elementwise operation (one op per element)
      profile.flopCount = numElements;

      // Memory footprint
      int64_t elemSize = inputType.getElementTypeBitWidth() / 8;
      int64_t totalInputSize = 0;
      for (auto input : inputs) {
        auto type = input.getType().cast<TensorType>();
        totalInputSize += type.getNumElements() * elemSize;
      }
      int64_t resultSize = resultType.getNumElements() * elemSize;
      profile.memoryFootprint = totalInputSize + resultSize;

      // Simple sequential access for elementwise ops
      for (auto input : inputs) {
        AccessInfo inputAccess;
        inputAccess.pattern = MemoryAccessPattern::SEQUENTIAL;
        inputAccess.accessCount = numElements;
        inputAccess.dataSize = input.getType().cast<TensorType>().getNumElements() * elemSize;
        inputAccess.intensity = static_cast<double>(profile.flopCount) / inputAccess.dataSize;
        inputAccess.isReusable = false;
        profile.inputAccess.push_back(inputAccess);
      }

      AccessInfo resultAccess;
      resultAccess.pattern = MemoryAccessPattern::SEQUENTIAL;
      resultAccess.accessCount = numElements;
      resultAccess.dataSize = resultSize;
      resultAccess.intensity = static_cast<double>(profile.flopCount) / resultSize;
      resultAccess.isReusable = false;
      profile.outputAccess.push_back(resultAccess);

      profile.computeIntensity = static_cast<double>(profile.flopCount) / profile.memoryFootprint;
      profile.hasDataReuse = false; // No reuse in elementwise ops
    }
  }

  return profile;
}

void TOSAGraphAnalysis::buildDataFlowGraph(func::FuncOp func) {
  func.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (auto defOp = operand.getDefiningOp()) {
        dataFlowGraph_[operand].push_back(op);
      }
    }
  });
}

void TOSAGraphAnalysis::computeCriticalPath() {
  // Simple critical path computation based on data dependencies
  for (auto &profile : opProfiles_) {
    criticalPath_[profile.op] = 1; // Base cost

    // Add dependencies
    for (auto operand : profile.op->getOperands()) {
      if (auto defOp = operand.getDefiningOp()) {
        auto it = criticalPath_.find(defOp);
        if (it != criticalPath_.end()) {
          criticalPath_[profile.op] = std::max(criticalPath_[profile.op], it->second + 1);
        }
      }
    }
  }
}

SmallVector<Operation*> TOSAGraphAnalysis::getMemoryBoundOps() const {
  SmallVector<Operation*> memoryBound;

  for (const auto &profile : opProfiles_) {
    // Operations with low compute intensity are memory-bound
    if (profile.computeIntensity < 10.0) { // Threshold: 10 FLOPs per byte
      memoryBound.push_back(profile.op);
    }
  }

  return memoryBound;
}

SmallVector<Operation*> TOSAGraphAnalysis::getComputeBoundOps() const {
  SmallVector<Operation*> computeBound;

  for (const auto &profile : opProfiles_) {
    // Operations with high compute intensity are compute-bound
    if (profile.computeIntensity >= 50.0) { // Threshold: 50 FLOPs per byte
      computeBound.push_back(profile.op);
    }
  }

  return computeBound;
}

//===----------------------------------------------------------------------===//
// OffloadCostModel Implementation
//===----------------------------------------------------------------------===//

double OffloadCostModel::estimateLocalCost(const TensorOpProfile &profile) const {
  // Simple cost model based on compute time + memory access time
  double computeTime = profile.flopCount / 1e12; // Assume 1 TFLOP/s compute
  double memoryTime = profile.memoryFootprint / (hardware.localBandwidth * 1e9);
  return std::max(computeTime, memoryTime);
}

double OffloadCostModel::estimateRemoteCost(const TensorOpProfile &profile, int memoryTier) const {
  double computeTime = profile.flopCount / 1e12; // Same compute capability

  double bandwidth, latency;
  switch (memoryTier) {
    case 0: // CXL attached
      bandwidth = hardware.cxlBandwidth * 1e9;
      latency = hardware.cxlLatency * 1e-9;
      break;
    case 1: // Far memory
      bandwidth = hardware.farBandwidth * 1e9;
      latency = hardware.farLatency * 1e-9;
      break;
    default:
      bandwidth = hardware.localBandwidth * 1e9;
      latency = hardware.localLatency * 1e-9;
  }

  double memoryTime = profile.memoryFootprint / bandwidth;
  double latencyOverhead = latency * profile.inputAccess.size(); // Access initiation overhead

  return std::max(computeTime, memoryTime + latencyOverhead);
}

int OffloadCostModel::selectOptimalTier(const TensorOpProfile &profile) const {
  double localCost = estimateLocalCost(profile);
  double cxlCost = estimateRemoteCost(profile, 0);
  double farCost = estimateRemoteCost(profile, 1);

  if (cxlCost < localCost && cxlCost <= farCost) {
    return 0; // CXL tier
  } else if (farCost < localCost) {
    return 1; // Far memory tier
  } else {
    return -1; // Stay local
  }
}

//===----------------------------------------------------------------------===//
// TOSAOffloadRecommender Implementation
//===----------------------------------------------------------------------===//

SmallVector<OffloadRecommendation> TOSAOffloadRecommender::recommendOffloads(func::FuncOp func) {
  SmallVector<OffloadRecommendation> recommendations;

  analysis_.analyzeFunction(func);

  // Initialize default cost model parameters
  costModel_.hardware.localBandwidth = 100.0;    // 100 GB/s
  costModel_.hardware.cxlBandwidth = 50.0;       // 50 GB/s
  costModel_.hardware.farBandwidth = 10.0;       // 10 GB/s
  costModel_.hardware.localLatency = 100.0;      // 100 ns
  costModel_.hardware.cxlLatency = 500.0;        // 500 ns
  costModel_.hardware.farLatency = 1000.0;       // 1 μs

  for (const auto &profile : analysis_.opProfiles_) {
    int optimalTier = costModel_.selectOptimalTier(profile);

    if (optimalTier >= 0) {
      OffloadRecommendation rec;
      rec.op = profile.op;
      rec.recommendedTier = optimalTier;
      rec.expectedSpeedup = costModel_.estimateLocalCost(profile) /
                           costModel_.estimateRemoteCost(profile, optimalTier);
      rec.rationale = explainRecommendation(rec);

      recommendations.push_back(rec);
    }
  }

  return recommendations;
}

StringRef TOSAOffloadRecommender::explainRecommendation(const OffloadRecommendation &rec) {
  // Simple explanation based on operation type and characteristics
  if (isa<tosa::MatMulOp>(rec.op)) {
    return "Large matrix multiplication benefits from CXL memory bandwidth";
  } else if (isa<tosa::Conv2DOp>(rec.op)) {
    return "Convolution with high memory footprint suits remote memory";
  } else if (isa<tosa::ReduceSumOp>(rec.op)) {
    return "Large reduction operation can utilize streaming remote access";
  } else {
    return "Operation shows favorable memory access patterns for offloading";
  }
}

} // namespace cira
} // namespace mlir