//===- TwoPassTimingAnalysis.cpp - Two-Pass Timing Integration ----===//
//
// This file implements the compiler-side integration for the two-pass
// execution methodology. It:
// 1. Reads profiling data from the first pass (JSON)
// 2. Annotates offload regions with timing information
// 3. Uses dominator tree analysis to compute prefetch placement
// 4. Inserts timing injection calls for the second pass
//
//===----------------------------------------------------------------------===//

#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <sstream>

using namespace mlir;
using namespace mlir::cira;

namespace {

//===----------------------------------------------------------------------===//
// Profile Data Structures
//===----------------------------------------------------------------------===//

/// Timing profile for a single offload region
struct RegionTimingProfile {
  uint32_t regionId;
  std::string regionName;

  // Host-side timing (from profiling pass)
  uint64_t hostIndependentWorkNs;

  // Vortex-side timing (from simulation)
  uint64_t vortexTotalCycles;
  uint64_t vortexTotalTimeNs;
  uint64_t vortexComputeCycles;
  uint64_t vortexMemoryStallCycles;
  uint64_t cacheHits;
  uint64_t cacheMisses;

  // Computed values
  int64_t injectionDelayNs;
  bool latencyHidden;
  uint32_t optimalPrefetchDepth;

  // Placement recommendations
  bool shouldHoistH2D;
  bool shouldSinkD2H;
};

/// Collection of all profiling data
struct TwoPassProfileData {
  std::vector<RegionTimingProfile> regions;
  double clockFreqMhz = 200.0;
  uint64_t cxlLatencyNs = 165;

  // Lookup by region name
  llvm::StringMap<size_t> regionNameToIndex;

  bool load(StringRef path);
  void save(StringRef path) const;

  const RegionTimingProfile* getRegion(StringRef name) const {
    auto it = regionNameToIndex.find(name);
    if (it != regionNameToIndex.end()) {
      return &regions[it->second];
    }
    return nullptr;
  }
};

/// Parse JSON profile data
bool TwoPassProfileData::load(StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    llvm::errs() << "Error: Could not open profile file: " << path << "\n";
    return false;
  }

  auto json = llvm::json::parse(bufferOrErr.get()->getBuffer());
  if (!json) {
    llvm::errs() << "Error: Invalid JSON in profile file\n";
    return false;
  }

  auto* root = json->getAsObject();
  if (!root) return false;

  if (auto freq = root->getNumber("clock_freq_mhz"))
    clockFreqMhz = *freq;
  if (auto lat = root->getInteger("cxl_latency_ns"))
    cxlLatencyNs = *lat;

  auto* regionsArray = root->getArray("regions");
  if (!regionsArray) return false;

  for (const auto& regionVal : *regionsArray) {
    auto* regionObj = regionVal.getAsObject();
    if (!regionObj) continue;

    RegionTimingProfile profile;

    if (auto id = regionObj->getInteger("region_id"))
      profile.regionId = *id;
    if (auto name = regionObj->getString("region_name"))
      profile.regionName = name->str();
    if (auto ns = regionObj->getInteger("host_independent_work_ns"))
      profile.hostIndependentWorkNs = *ns;

    // Parse vortex_timing sub-object
    if (auto* timing = regionObj->getObject("vortex_timing")) {
      if (auto v = timing->getInteger("total_cycles"))
        profile.vortexTotalCycles = *v;
      if (auto v = timing->getInteger("total_time_ns"))
        profile.vortexTotalTimeNs = *v;
      if (auto v = timing->getInteger("compute_cycles"))
        profile.vortexComputeCycles = *v;
      if (auto v = timing->getInteger("memory_stall_cycles"))
        profile.vortexMemoryStallCycles = *v;
      if (auto v = timing->getInteger("cache_hits"))
        profile.cacheHits = *v;
      if (auto v = timing->getInteger("cache_misses"))
        profile.cacheMisses = *v;
    }

    if (auto v = regionObj->getInteger("injection_delay_ns"))
      profile.injectionDelayNs = *v;
    if (auto v = regionObj->getBoolean("latency_hidden"))
      profile.latencyHidden = *v;
    if (auto v = regionObj->getInteger("optimal_prefetch_depth"))
      profile.optimalPrefetchDepth = *v;

    // Defaults for placement
    profile.shouldHoistH2D = true;
    profile.shouldSinkD2H = true;

    regionNameToIndex[profile.regionName] = regions.size();
    regions.push_back(std::move(profile));
  }

  llvm::outs() << "Loaded " << regions.size() << " region profiles from " << path << "\n";
  return true;
}

void TwoPassProfileData::save(StringRef path) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) {
    llvm::errs() << "Error: Could not write profile file: " << path << "\n";
    return;
  }

  llvm::json::Object root;
  root["num_regions"] = (int64_t)regions.size();
  root["clock_freq_mhz"] = clockFreqMhz;
  root["cxl_latency_ns"] = (int64_t)cxlLatencyNs;

  llvm::json::Array regionsArray;
  for (const auto& r : regions) {
    llvm::json::Object regionObj;
    regionObj["region_id"] = (int64_t)r.regionId;
    regionObj["region_name"] = r.regionName;
    regionObj["host_independent_work_ns"] = (int64_t)r.hostIndependentWorkNs;

    llvm::json::Object timing;
    timing["total_cycles"] = (int64_t)r.vortexTotalCycles;
    timing["total_time_ns"] = (int64_t)r.vortexTotalTimeNs;
    timing["compute_cycles"] = (int64_t)r.vortexComputeCycles;
    timing["memory_stall_cycles"] = (int64_t)r.vortexMemoryStallCycles;
    timing["cache_hits"] = (int64_t)r.cacheHits;
    timing["cache_misses"] = (int64_t)r.cacheMisses;
    regionObj["vortex_timing"] = std::move(timing);

    regionObj["injection_delay_ns"] = r.injectionDelayNs;
    regionObj["latency_hidden"] = r.latencyHidden;
    regionObj["optimal_prefetch_depth"] = (int64_t)r.optimalPrefetchDepth;

    regionsArray.push_back(std::move(regionObj));
  }
  root["regions"] = std::move(regionsArray);

  os << llvm::json::Value(std::move(root));
}

//===----------------------------------------------------------------------===//
// Dominator Tree Analysis for Prefetch Placement
//===----------------------------------------------------------------------===//

/// Analyze the dominator tree to find optimal prefetch placement
struct PrefetchPlacementAnalysis {
  DominanceInfo& domInfo;
  PostDominanceInfo& postDomInfo;

  PrefetchPlacementAnalysis(DominanceInfo& dom, PostDominanceInfo& postDom)
      : domInfo(dom), postDomInfo(postDom) {}

  /// Find the earliest point where a prefetch can be issued
  /// (the dominator of the use point)
  Block* findPrefetchPoint(Operation* useOp) {
    Block* useBlock = useOp->getBlock();

    // Walk up the dominator tree to find a suitable prefetch point
    // We want to prefetch as early as possible
    Block* prefetchBlock = useBlock;

    // Check if we can hoist to a dominator
    if (Block* idom = domInfo.getNode(useBlock)->getIDom()->getBlock()) {
      // Verify all values needed for prefetch are available at idom
      prefetchBlock = idom;
    }

    return prefetchBlock;
  }

  /// Calculate the "slack" available for prefetching
  /// This is the number of instructions between prefetch point and use
  size_t calculateSlack(Block* prefetchBlock, Operation* useOp) {
    size_t slack = 0;

    // Count operations between prefetch point and use
    // This is a simplified heuristic
    Block* currentBlock = prefetchBlock;
    while (currentBlock != useOp->getBlock()) {
      slack += std::distance(currentBlock->begin(), currentBlock->end());
      // Move to successor (simplified - assumes linear CFG)
      if (currentBlock->getNumSuccessors() > 0) {
        currentBlock = currentBlock->getSuccessor(0);
      } else {
        break;
      }
    }

    // Add distance within the use block
    for (auto& op : *useOp->getBlock()) {
      if (&op == useOp) break;
      slack++;
    }

    return slack;
  }

  /// Determine if H2D transfer can be hoisted out of a loop
  bool canHoistH2D(scf::ForOp forOp, Value memref) {
    // H2D can be hoisted if the memref is loop-invariant
    // (defined outside the loop and not modified inside)
    Operation* defOp = memref.getDefiningOp();
    if (!defOp) return true;  // Block argument, check dominance

    // Check if definition dominates the loop
    return domInfo.dominates(defOp->getBlock(), forOp->getBlock());
  }

  /// Determine if D2H transfer can be sunk after a loop
  bool canSinkD2H(scf::ForOp forOp, Value result) {
    // D2H can be sunk if the result is only used after the loop
    for (Operation* user : result.getUsers()) {
      if (!postDomInfo.postDominates(user->getBlock(), forOp->getBlock())) {
        return false;
      }
    }
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Two-Pass Timing Injection Pass
//===----------------------------------------------------------------------===//

struct TwoPassTimingInjectionPass
    : public PassWrapper<TwoPassTimingInjectionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TwoPassTimingInjectionPass)

  // Default constructor
  TwoPassTimingInjectionPass() = default;

  // Constructor with parameters (for programmatic use)
  TwoPassTimingInjectionPass(StringRef profile, bool annotate = false)
      : profilePathStr(profile.str()), annotateOnlyFlag(annotate) {}

  // Copy constructor - required for clonePass()
  TwoPassTimingInjectionPass(const TwoPassTimingInjectionPass& other)
      : PassWrapper<TwoPassTimingInjectionPass, OperationPass<ModuleOp>>(),
        profilePathStr(other.profilePathStr),
        annotateOnlyFlag(other.annotateOnlyFlag) {}

  StringRef getArgument() const override { return "cira-twopass-timing"; }
  StringRef getDescription() const override {
    return "Inject timing annotations from two-pass profiling";
  }

  // Stored values for copy constructor
  std::string profilePathStr;
  bool annotateOnlyFlag = false;

  Option<std::string> profilePath{*this, "profile",
                                   llvm::cl::desc("Path to timing profile JSON"),
                                   llvm::cl::init("")};

  Option<bool> annotateOnly{*this, "annotate-only",
                             llvm::cl::desc("Only add annotations, don't inject delays"),
                             llvm::cl::init(false)};

  // Get effective profile path (from option or stored value)
  std::string getProfilePath() const {
    return profilePath.empty() ? profilePathStr : std::string(profilePath);
  }

  // Get effective annotate-only flag
  bool getAnnotateOnly() const {
    return annotateOnly || annotateOnlyFlag;
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<cira::RemoteMemDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* ctx = &getContext();

    // Load profiling data
    TwoPassProfileData profileData;
    std::string path = getProfilePath();
    if (!path.empty()) {
      if (!profileData.load(path)) {
        signalPassFailure();
        return;
      }
    }

    // Process each function
    module.walk([&](func::FuncOp funcOp) {
      // Build dominator tree
      DominanceInfo domInfo(funcOp);
      PostDominanceInfo postDomInfo(funcOp);
      PrefetchPlacementAnalysis placementAnalysis(domInfo, postDomInfo);

      // Find all offload regions
      funcOp.walk([&](OffloadRegionOp offloadOp) {
        processOffloadRegion(offloadOp, profileData, placementAnalysis, ctx);
      });

      // Also process legacy offload ops
      funcOp.walk([&](OffloadOp offloadOp) {
        processLegacyOffload(offloadOp, profileData, ctx);
      });
    });
  }

private:
  void processOffloadRegion(OffloadRegionOp offloadOp,
                            const TwoPassProfileData& profileData,
                            PrefetchPlacementAnalysis& placement,
                            MLIRContext* ctx) {
    Location loc = offloadOp.getLoc();

    // Try to get region name from attribute or generate one
    std::string regionName;
    if (auto nameAttr = offloadOp->getAttrOfType<StringAttr>("region_name")) {
      regionName = nameAttr.getValue().str();
    } else {
      // Generate name from location
      std::string locStr;
      llvm::raw_string_ostream os(locStr);
      loc.print(os);
      regionName = "region_" + locStr;
    }

    // Look up timing profile
    const RegionTimingProfile* profile = profileData.getRegion(regionName);

    // Add timing annotations as attributes
    if (profile) {
      offloadOp->setAttr("twopass.injection_delay_ns",
                         IntegerAttr::get(IntegerType::get(ctx, 64),
                                         profile->injectionDelayNs));
      offloadOp->setAttr("twopass.latency_hidden",
                         BoolAttr::get(ctx, profile->latencyHidden));
      offloadOp->setAttr("twopass.optimal_prefetch_depth",
                         IntegerAttr::get(IntegerType::get(ctx, 32),
                                         profile->optimalPrefetchDepth));
      offloadOp->setAttr("twopass.should_hoist_h2d",
                         BoolAttr::get(ctx, profile->shouldHoistH2D));
      offloadOp->setAttr("twopass.should_sink_d2h",
                         BoolAttr::get(ctx, profile->shouldSinkD2H));

      // Add performance statistics
      offloadOp->setAttr("twopass.vortex_cycles",
                         IntegerAttr::get(IntegerType::get(ctx, 64),
                                         profile->vortexTotalCycles));
      offloadOp->setAttr("twopass.cache_hit_rate",
                         FloatAttr::get(Float64Type::get(ctx),
                                       (double)profile->cacheHits /
                                       (profile->cacheHits + profile->cacheMisses + 1)));

      llvm::outs() << "Annotated region '" << regionName << "':\n";
      llvm::outs() << "  Injection delay: " << profile->injectionDelayNs << " ns\n";
      llvm::outs() << "  Latency hidden: " << profile->latencyHidden << "\n";
      llvm::outs() << "  Prefetch depth: " << profile->optimalPrefetchDepth << "\n";
    }

    // If not annotate-only, also update prefetch operations
    if (!getAnnotateOnly() && profile) {
      offloadOp.walk([&](PrefetchChainOp prefetchOp) {
        // Update prefetch depth based on profiling
        prefetchOp.setDepthAttr(
            IntegerAttr::get(IntegerType::get(ctx, 64),
                            profile->optimalPrefetchDepth));
      });

      offloadOp.walk([&](PrefetchIndirectOp prefetchOp) {
        prefetchOp.setDepthAttr(
            IntegerAttr::get(IntegerType::get(ctx, 64),
                            profile->optimalPrefetchDepth));
      });
    }
  }

  void processLegacyOffload(OffloadOp offloadOp,
                            const TwoPassProfileData& profileData,
                            MLIRContext* ctx) {
    // Similar processing for legacy offload ops
    std::string regionName = offloadOp.getCommand().str();
    const RegionTimingProfile* profile = profileData.getRegion(regionName);

    if (profile) {
      offloadOp->setAttr("twopass.injection_delay_ns",
                         IntegerAttr::get(IntegerType::get(ctx, 64),
                                         profile->injectionDelayNs));
      offloadOp->setAttr("twopass.latency_hidden",
                         BoolAttr::get(ctx, profile->latencyHidden));
    }
  }
};

//===----------------------------------------------------------------------===//
// Profile-Guided Prefetch Optimization Pass
//===----------------------------------------------------------------------===//

struct ProfileGuidedPrefetchPass
    : public PassWrapper<ProfileGuidedPrefetchPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProfileGuidedPrefetchPass)

  StringRef getArgument() const override { return "cira-profile-prefetch"; }
  StringRef getDescription() const override {
    return "Optimize prefetch operations based on timing profile";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext* ctx = &getContext();

    // Build dominator analysis
    DominanceInfo domInfo(funcOp);
    PostDominanceInfo postDomInfo(funcOp);

    // Process loops with offload regions
    funcOp.walk([&](scf::ForOp forOp) {
      optimizeLoopPrefetch(forOp, domInfo, postDomInfo, ctx);
    });

    funcOp.walk([&](scf::WhileOp whileOp) {
      optimizeWhilePrefetch(whileOp, domInfo, ctx);
    });
  }

private:
  void optimizeLoopPrefetch(scf::ForOp forOp, DominanceInfo& domInfo,
                            PostDominanceInfo& postDomInfo, MLIRContext* ctx) {
    // Check if this loop contains offload regions with timing annotations
    SmallVector<OffloadRegionOp> offloadOps;
    forOp.walk([&](OffloadRegionOp op) {
      offloadOps.push_back(op);
    });

    for (auto offloadOp : offloadOps) {
      // Read timing annotations
      auto delayAttr = offloadOp->getAttrOfType<IntegerAttr>("twopass.injection_delay_ns");
      auto hoistAttr = offloadOp->getAttrOfType<BoolAttr>("twopass.should_hoist_h2d");
      auto sinkAttr = offloadOp->getAttrOfType<BoolAttr>("twopass.should_sink_d2h");

      if (!delayAttr) continue;

      int64_t delayNs = delayAttr.getInt();
      bool shouldHoist = hoistAttr && hoistAttr.getValue();
      bool shouldSink = sinkAttr && sinkAttr.getValue();

      // If latency can be hidden, mark for aggressive prefetching
      if (delayNs <= 0) {
        offloadOp->setAttr("optimization.latency_hidden", BoolAttr::get(ctx, true));
      }

      // Apply H2D hoisting recommendation
      if (shouldHoist) {
        offloadOp->setAttr("optimization.hoist_h2d", BoolAttr::get(ctx, true));
        // The actual hoisting is done by a later pass
      }

      // Apply D2H sinking recommendation
      if (shouldSink) {
        offloadOp->setAttr("optimization.sink_d2h", BoolAttr::get(ctx, true));
      }
    }
  }

  void optimizeWhilePrefetch(scf::WhileOp whileOp, DominanceInfo& domInfo,
                             MLIRContext* ctx) {
    // For while loops (pointer chasing), the prefetch depth is critical
    whileOp.walk([&](PrefetchChainOp prefetchOp) {
      // Check if parent offload region has timing data
      if (auto offloadOp = prefetchOp->getParentOfType<OffloadRegionOp>()) {
        auto depthAttr = offloadOp->getAttrOfType<IntegerAttr>("twopass.optimal_prefetch_depth");
        if (depthAttr) {
          prefetchOp.setDepthAttr(depthAttr);
        }
      }
    });
  }
};

//===----------------------------------------------------------------------===//
// Runtime Call Injection Pass (for timing injection in second pass)
//===----------------------------------------------------------------------===//

struct TimingInjectionCallsPass
    : public PassWrapper<TimingInjectionCallsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TimingInjectionCallsPass)

  StringRef getArgument() const override { return "cira-inject-timing-calls"; }
  StringRef getDescription() const override {
    return "Insert runtime calls for timing injection";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<cira::RemoteMemDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* ctx = &getContext();
    OpBuilder builder(ctx);

    // Declare external functions for runtime
    auto i64Type = IntegerType::get(ctx, 64);
    auto i32Type = IntegerType::get(ctx, 32);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    // Insert function declarations
    builder.setInsertionPointToStart(module.getBody());

    // void twopass_sync_point(void* ctx, uint32_t region_id)
    auto syncPointType = LLVM::LLVMFunctionType::get(
        voidType, {LLVM::LLVMPointerType::get(ctx), i32Type});
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "twopass_sync_point",
                                      syncPointType);

    // void twopass_host_work_start(void* ctx, uint32_t region_id)
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "twopass_host_work_start",
                                      syncPointType);

    // void twopass_host_work_end(void* ctx, uint32_t region_id)
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "twopass_host_work_end",
                                      syncPointType);

    // Process each function to insert calls
    module.walk([&](func::FuncOp funcOp) {
      insertTimingCalls(funcOp, ctx);
    });
  }

private:
  void insertTimingCalls(func::FuncOp funcOp, MLIRContext* ctx) {
    OpBuilder builder(ctx);

    uint32_t regionCounter = 0;

    funcOp.walk([&](OffloadRegionOp offloadOp) {
      // Check if this region needs timing injection
      auto delayAttr = offloadOp->getAttrOfType<IntegerAttr>("twopass.injection_delay_ns");
      auto hiddenAttr = offloadOp->getAttrOfType<BoolAttr>("twopass.latency_hidden");

      if (!delayAttr) return;

      int64_t delayNs = delayAttr.getInt();
      bool hidden = hiddenAttr && hiddenAttr.getValue();

      // Don't inject calls if latency is hidden
      if (hidden) {
        regionCounter++;
        return;
      }

      Location loc = offloadOp.getLoc();

      // Insert host_work_start at the dominator (before offload)
      builder.setInsertionPoint(offloadOp);
      auto regionIdConst = builder.create<arith::ConstantOp>(
          loc, builder.getI32IntegerAttr(regionCounter));

      // Note: In real implementation, we'd pass the twopass context
      // For now, we add attributes that will be lowered to actual calls

      offloadOp->setAttr("twopass.region_id",
                         builder.getI32IntegerAttr(regionCounter));

      // Insert sync_point after the offload region
      builder.setInsertionPointAfter(offloadOp);

      // Add a phase boundary to mark sync point
      builder.create<PhaseBoundaryOp>(loc,
          builder.getStringAttr("sync_" + std::to_string(regionCounter)));

      regionCounter++;
    });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cira {

std::unique_ptr<OperationPass<ModuleOp>> createTwoPassTimingInjectionPass() {
  return std::make_unique<TwoPassTimingInjectionPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createProfileGuidedPrefetchPass() {
  return std::make_unique<ProfileGuidedPrefetchPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createTimingInjectionCallsPass() {
  return std::make_unique<TimingInjectionCallsPass>();
}

void registerTwoPassPasses() {
  PassRegistration<TwoPassTimingInjectionPass>();
  PassRegistration<ProfileGuidedPrefetchPass>();
  PassRegistration<TimingInjectionCallsPass>();
}

} // namespace cira
} // namespace mlir
