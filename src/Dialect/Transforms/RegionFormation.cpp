//===- RegionFormation.cpp - Algorithm 1: static region formation ---------===//
//
// Walks every memory operation that touches CXL-resident data, forms a
// backward slice rooted at it, prunes what the device cannot run, and outlines
// the survivor into a `cira.offload` region followed by a `cira.barrier`.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "Dialect/Transforms/BackwardSlice.h"
#include "Dialect/Transforms/Passes.h"
#include "Dialect/TwoPassTimingAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cira-region-formation"

namespace mlir {
#define GEN_PASS_DEF_CIRAREGIONFORMATION
#include "Dialect/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::cira;

namespace {

/// Recover the originating source file so regions stay attributable after
/// several translation units have been linked into one module.
StringRef findSourceFile(Location loc) {
    if (auto file = dyn_cast<FileLineColLoc>(loc))
        return file.getFilename().getValue();
    if (auto fused = dyn_cast<FusedLoc>(loc))
        for (Location sub : fused.getLocations())
            if (StringRef name = findSourceFile(sub); !name.empty())
                return name;
    if (auto named = dyn_cast<NameLoc>(loc))
        return findSourceFile(named.getChildLoc());
    if (auto call = dyn_cast<CallSiteLoc>(loc))
        return findSourceFile(call.getCallee());
    return {};
}

struct RegionCandidate {
    SmallVector<Operation *> roots;
    SmallVector<Operation *> ops;
    SetVector<Value> liveIns;
    Region *parentRegion = nullptr;
    Block *block = nullptr;
    unsigned chainDepth = 0;
};

class CiraRegionFormationPass : public impl::CiraRegionFormationBase<CiraRegionFormationPass> {
public:
    using impl::CiraRegionFormationBase<CiraRegionFormationPass>::CiraRegionFormationBase;

    void runOnOperation() override {
        ModuleOp module = getOperation();
        unsigned regionId = 0;
        unsigned numFuncs = 0, numMemOps = 0, numRoots = 0;

        for (auto func : module.getOps<FunctionOpInterface>()) {
            if (func.isExternal())
                continue;
            ++numFuncs;

            SmallVector<Operation *> roots;
            func.walk([&](Operation *op) {
                if (op->getParentOfType<OffloadRegionOp>())
                    return;
                if (!getAddressOperands(op).empty())
                    ++numMemOps;
                if (isRemoteMemoryAccess(op, allowLegacyHeapRemote))
                    roots.push_back(op);
            });
            numRoots += roots.size();

            // Phase 1: form candidates without mutating the IR.
            SmallVector<RegionCandidate> candidates;
            for (Operation *root : roots) {
                RegionCandidate candidate;
                if (buildCandidate(root, candidate))
                    candidates.push_back(std::move(candidate));
            }

            // Phase 2: greedily union adjacent compatible candidates.
            DominanceInfo dominance(func);
            SmallVector<RegionCandidate> merged;
            for (RegionCandidate &candidate : candidates) {
                if (!findSubmitPoint(candidate, dominance))
                    continue;
                if (!merged.empty() && canMerge(merged.back(), candidate)) {
                    RegionCandidate combined = merged.back();
                    mergeInto(combined, candidate);
                    if (findSubmitPoint(combined, dominance)) {
                        merged.back() = std::move(combined);
                        continue;
                    }
                }
                merged.push_back(std::move(candidate));
            }

            // Phase 3: outline only after all grouping/placement decisions.
            SmallVector<Operation *> submitPoints;
            submitPoints.reserve(merged.size());
            for (RegionCandidate &candidate : merged)
                submitPoints.push_back(findSubmitPoint(candidate, dominance));
            for (auto [candidate, submitPoint] : llvm::zip(merged, submitPoints))
                if (submitPoint)
                    outlineAsDeviceKernel(candidate, submitPoint, regionId++);
        }

        LLVM_DEBUG(llvm::dbgs() << "STATS funcs=" << numFuncs << " memops=" << numMemOps << " roots=" << numRoots
                                << " regions=" << regionId << "\n");
        if (regionId == 0)
            LLVM_DEBUG(llvm::dbgs() << "no offloadable region found\n");
    }

private:
    bool buildCandidate(Operation *root, RegionCandidate &candidate) {
        // Device writes are outside this read-only milestone.
        if (!isMemoryRead(root))
            return false;

        SliceOptions options;
        options.maxOps = maxSliceOps;
        options.followLoads = followDependentLoads;

        BackwardSliceResult slice = computeBackwardSlice(root, options);
        pruneUnsupported(slice);

        if (!isDeviceLegal(slice)) {
            LLVM_DEBUG(llvm::dbgs() << "slice not device-legal at " << *root << "\n");
            return false;
        }
        if (!hasRemoteDependence(slice)) {
            LLVM_DEBUG(llvm::dbgs() << "slice has no remote dependence at " << *root << "\n");
            return false;
        }
        if (requireProfitable && !OffloadCostModelParams::shouldOffload(slice.chainDepth)) {
            LLVM_DEBUG(llvm::dbgs() << "slice not profitable (depth " << slice.chainDepth << ") at " << *root << "\n");
            return false;
        }

        candidate.roots.push_back(root);
        candidate.parentRegion = root->getParentRegion();
        candidate.block = root->getBlock();
        candidate.chainDepth = slice.chainDepth;
        for (Operation *op : slice.ops)
            if (op->getBlock() == candidate.block)
                candidate.ops.push_back(op);
        if (candidate.ops.empty() || !llvm::any_of(candidate.ops, isMemoryRead))
            return false;
        candidate.liveIns = computeSliceLiveIns(candidate.ops, candidate.roots);
        return true;
    }

    static bool hasInterveningHostEffect(const RegionCandidate &left, const RegionCandidate &right) {
        Operation *from = left.roots.back();
        Operation *to = right.roots.front();
        if (from->getBlock() != to->getBlock() || !from->isBeforeInBlock(to))
            return true;
        for (Operation *op = from->getNextNode(); op && op != to; op = op->getNextNode())
            if (isHostOnlyOp(op))
                return true;
        return false;
    }

    static bool canMerge(const RegionCandidate &left, const RegionCandidate &right) {
        if (left.parentRegion != right.parentRegion || left.block != right.block)
            return false;
        if (hasInterveningHostEffect(left, right))
            return false;
        return llvm::all_of(left.ops, isDeviceLegalOp) && llvm::all_of(right.ops, isDeviceLegalOp) &&
               llvm::all_of(left.roots, isMemoryRead) && llvm::all_of(right.roots, isMemoryRead);
    }

    static void mergeInto(RegionCandidate &destination, const RegionCandidate &source) {
        SetVector<Operation *> unionOps(destination.ops.begin(), destination.ops.end());
        unionOps.insert(source.ops.begin(), source.ops.end());
        destination.ops.assign(unionOps.begin(), unionOps.end());
        llvm::sort(destination.ops, [](Operation *left, Operation *right) { return left->isBeforeInBlock(right); });
        destination.roots.append(source.roots.begin(), source.roots.end());
        destination.chainDepth = std::max(destination.chainDepth, source.chainDepth);
        destination.liveIns = computeSliceLiveIns(destination.ops, destination.roots);
    }

    static Operation *findFirstPreparedOp(const RegionCandidate &candidate) {
        Operation *first = candidate.roots.front();
        for (Operation *op : candidate.ops)
            if (op->isBeforeInBlock(first))
                first = op;
        return first;
    }

    static Operation *findSubmitPoint(const RegionCandidate &candidate, DominanceInfo &dominance) {
        Operation *firstPrepared = findFirstPreparedOp(candidate);
        Operation *latestDefinition = nullptr;
        for (Value liveIn : candidate.liveIns) {
            Operation *def = liveIn.getDefiningOp();
            if (!def || def->getBlock() != candidate.block)
                continue;
            if (!latestDefinition || latestDefinition->isBeforeInBlock(def))
                latestDefinition = def;
        }

        Operation *submitPoint = latestDefinition ? latestDefinition->getNextNode() : &candidate.block->front();
        if (!submitPoint || (submitPoint != firstPrepared && firstPrepared->isBeforeInBlock(submitPoint)))
            return nullptr;
        for (Value liveIn : candidate.liveIns)
            if (!dominance.dominates(liveIn, submitPoint))
                return nullptr;
        return submitPoint;
    }

    /// Clone the merged address-production slice and read-only roots. The host
    /// operations stay authoritative and consume the prepared lines after one
    /// wait at the earliest root.
    void outlineAsDeviceKernel(const RegionCandidate &candidate, Operation *submitPoint, unsigned regionId) {
        Operation *waitPoint = candidate.roots.front();
        OpBuilder builder(submitPoint);
        Location loc = waitPoint->getLoc();

        SmallVector<Value> liveIns(candidate.liveIns.begin(), candidate.liveIns.end());
        SmallVector<Type> liveInTypes;
        for (Value v : liveIns)
            liveInTypes.push_back(v.getType());

        builder.setInsertionPoint(submitPoint);

        auto offloadOp =
            builder.create<OffloadRegionOp>(loc, /*results=*/TypeRange{}, /*target=*/SymbolRefAttr(), liveIns);
        offloadOp->setAttr("cira.region_id", builder.getI32IntegerAttr(regionId));
        offloadOp->setAttr("cira.chain_depth", builder.getI32IntegerAttr(candidate.chainDepth));
        offloadOp->setAttr("cira.slice_size", builder.getI32IntegerAttr(candidate.ops.size()));
        offloadOp->setAttr("cira.root_count", builder.getI32IntegerAttr(candidate.roots.size()));
        unsigned overlapOps = 0;
        for (Operation *op = submitPoint; op && op != waitPoint; op = op->getNextNode())
            ++overlapOps;
        offloadOp->setAttr("cira.overlap_ops", builder.getI32IntegerAttr(overlapOps));
        if (StringRef source = findSourceFile(loc); !source.empty())
            offloadOp->setAttr("cira.source", builder.getStringAttr(source));

        SmallVector<Location> argLocs(liveIns.size(), loc);
        Block *body = builder.createBlock(&offloadOp.getBody(), {}, liveInTypes, argLocs);
        builder.setInsertionPointToStart(body);

        IRMapping mapping;
        for (auto [outer, arg] : llvm::zip(liveIns, body->getArguments()))
            mapping.map(outer, arg);
        rematerializeConstants(candidate, builder, mapping);
        for (Operation *op : candidate.ops)
            builder.clone(*op, mapping);
        for (Operation *root : candidate.roots)
            if (!llvm::is_contained(candidate.ops, root))
                builder.clone(*root, mapping);

        builder.setInsertionPoint(waitPoint);
        builder.create<BarrierOp>(loc);
    }

    /// Constants are boundaries of the slice but are cheaper to recreate on the
    /// device than to ship as operands.
    void rematerializeConstants(const RegionCandidate &candidate, OpBuilder &builder, IRMapping &mapping) {
        SetVector<Operation *> constants;
        auto collect = [&](ValueRange values) {
            for (Value operand : values) {
                Operation *def = operand.getDefiningOp();
                if (def && !mapping.contains(operand) && matchPattern(operand, m_Constant()))
                    constants.insert(def);
            }
        };
        for (Operation *op : candidate.ops)
            collect(op->getOperands());
        for (Operation *root : candidate.roots)
            collect(root->getOperands());

        for (Operation *constant : constants)
            builder.clone(*constant, mapping);
    }
};

} // namespace
