// CIRA JIT engine — ORC LLJIT-backed specializer. See cira_jit_engine.h.

#include "cira_jit_engine.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstring>
#include <functional>

namespace cira {

namespace {

std::atomic<bool> g_targetInited{false};

// Fold one i32 sentinel global to a constant initializer + internal linkage,
// so the optimizer can constant-propagate uses across the IR.
void patchI32(llvm::Module& M, const char* name, uint32_t value) {
    llvm::GlobalVariable* g = M.getGlobalVariable(name, /*AllowInternal=*/true);
    if (!g) return;
    auto* ty = llvm::Type::getInt32Ty(M.getContext());
    g->setConstant(true);
    g->setInitializer(llvm::ConstantInt::get(ty, value, /*isSigned=*/false));
    g->setLinkage(llvm::GlobalValue::InternalLinkage);
}

void patchF32(llvm::Module& M, const char* name, float value) {
    llvm::GlobalVariable* g = M.getGlobalVariable(name, /*AllowInternal=*/true);
    if (!g) return;
    auto* ty = llvm::Type::getFloatTy(M.getContext());
    g->setConstant(true);
    g->setInitializer(llvm::ConstantFP::get(ty, (double)value));
    g->setLinkage(llvm::GlobalValue::InternalLinkage);
}

uint64_t mix64(uint64_t x) {
    x ^= x >> 33;  x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;  x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

} // namespace

bool CiraJitEngine::initializeNativeTarget() {
    bool expected = false;
    if (!g_targetInited.compare_exchange_strong(expected, true)) return true;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return true;
}

CiraJitEngine::CiraJitEngine() = default;
CiraJitEngine::~CiraJitEngine() = default;

uint64_t CiraJitEngine::fingerprint(const cira_jit_decision_t& d) {
    // Quantize host_device_split to 1/256 so trivially-different floats
    // collapse onto the same compiled code.
    uint64_t s = (uint64_t)(d.host_device_split * 256.0f) & 0xff;
    uint64_t v = ((uint64_t)d.batch_size        & 0xffff)
               | (((uint64_t)d.traversal_depth   & 0xffff) << 16)
               | (((uint64_t)d.pipeline_distance & 0xffff) << 32)
               | ((s & 0xff) << 48)
               | (((uint64_t)(d.should_offload ? 1u : 0u)) << 56);
    return mix64(v);
}

void CiraJitEngine::patchAndOptimize(llvm::Module&              M,
                                     const cira_jit_decision_t& d) {
    patchI32(M, kSentinelBatchSize,        d.batch_size);
    patchI32(M, kSentinelTraversalDepth,   d.traversal_depth);
    patchI32(M, kSentinelPipelineDistance, d.pipeline_distance);
    patchF32(M, kSentinelHostDeviceSplit,  d.host_device_split);

    // Run the standard O3 pipeline so constant-prop / unrolling / branch
    // folding can collapse code that switches on the sentinel values.
    llvm::PassBuilder PB;
    llvm::LoopAnalysisManager     LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager    CGAM;
    llvm::ModuleAnalysisManager   MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    auto MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    MPM.run(M, MAM);
}

bool CiraJitEngine::ensureLLJIT() {
    if (jit_) return true;
    initializeNativeTarget();
    auto J = llvm::orc::LLJITBuilder().create();
    if (!J) {
        llvm::errs() << "[cira-jit] LLJIT create failed: "
                     << llvm::toString(J.takeError()) << "\n";
        return false;
    }
    jit_ = std::move(*J);
    // ctx_ is no longer cached at the engine level — modern ORC requires a
    // fresh LLVMContext per ThreadSafeModule. We construct one inside each
    // specialize() call and transfer ownership to LLJIT.
    return true;
}

CiraJitFn CiraJitEngine::specialize(const std::string&          bitcodePath,
                                    const std::string&          kernelName,
                                    const cira_jit_decision_t&  decision) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ensureLLJIT()) return nullptr;

    CacheKey key{kernelName, bitcodePath, fingerprint(decision)};
    if (auto it = cache_.find(key); it != cache_.end()) return it->second;

    auto buf = llvm::MemoryBuffer::getFile(bitcodePath);
    if (!buf) {
        llvm::errs() << "[cira-jit] read failed: " << bitcodePath << ": "
                     << buf.getError().message() << "\n";
        return nullptr;
    }
    // Each specialization gets a fresh LLVMContext that we hand to ORC.
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto modOrErr = llvm::parseBitcodeFile((*buf)->getMemBufferRef(), *ctx);
    if (!modOrErr) {
        llvm::errs() << "[cira-jit] parseBitcodeFile failed: "
                     << llvm::toString(modOrErr.takeError()) << "\n";
        return nullptr;
    }
    std::unique_ptr<llvm::Module> M = std::move(*modOrErr);
    patchAndOptimize(*M, decision);

    if (auto err = jit_->addIRModule(
            llvm::orc::ThreadSafeModule(std::move(M), std::move(ctx)))) {
        llvm::errs() << "[cira-jit] addIRModule failed: "
                     << llvm::toString(std::move(err)) << "\n";
        return nullptr;
    }

    auto sym = jit_->lookup(kernelName);
    if (!sym) {
        llvm::errs() << "[cira-jit] lookup '" << kernelName << "' failed: "
                     << llvm::toString(sym.takeError()) << "\n";
        return nullptr;
    }
    auto fn = reinterpret_cast<CiraJitFn>(sym->getValue());
    cache_.emplace(std::move(key), fn);
    return fn;
}

CiraJitFn CiraJitEngine::specializeFromIR(const std::string&         irText,
                                          const std::string&         kernelName,
                                          const cira_jit_decision_t& decision) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ensureLLJIT()) return nullptr;

    // Use a content hash as the cache "source" so identical IR text shares
    // a cache entry.
    uint64_t srcHash = std::hash<std::string>{}(irText);
    char srcKey[32];
    std::snprintf(srcKey, sizeof(srcKey), "ir@%016llx",
                  (unsigned long long)srcHash);

    CacheKey key{kernelName, srcKey, fingerprint(decision)};
    if (auto it = cache_.find(key); it != cache_.end()) return it->second;

    auto ctx = std::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic err;
    auto buf = llvm::MemoryBuffer::getMemBuffer(irText);
    auto M = llvm::parseIR(buf->getMemBufferRef(), err, *ctx);
    if (!M) {
        llvm::errs() << "[cira-jit] parseIR failed: ";
        err.print("cira-jit", llvm::errs());
        return nullptr;
    }
    patchAndOptimize(*M, decision);

    if (auto e = jit_->addIRModule(
            llvm::orc::ThreadSafeModule(std::move(M), std::move(ctx)))) {
        llvm::errs() << "[cira-jit] addIRModule failed: "
                     << llvm::toString(std::move(e)) << "\n";
        return nullptr;
    }
    auto sym = jit_->lookup(kernelName);
    if (!sym) {
        llvm::errs() << "[cira-jit] lookup '" << kernelName << "' failed: "
                     << llvm::toString(sym.takeError()) << "\n";
        return nullptr;
    }
    auto fn = reinterpret_cast<CiraJitFn>(sym->getValue());
    cache_.emplace(std::move(key), fn);
    return fn;
}

void CiraJitEngine::resetCache() {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.clear();
    jit_.reset();
}

} // namespace cira
