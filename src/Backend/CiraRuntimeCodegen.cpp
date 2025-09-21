#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace cira {

// Runtime code generator that emits C++ code using the Cira runtime
class CiraRuntimeCodeGenerator {
private:
    llvm::raw_ostream &os;
    std::string indent;
    int indentLevel = 0;

    // Track runtime initialization
    bool runtimeInitialized = false;

    // Variable name generation
    int varCounter = 0;
    std::map<Value, std::string> valueNames;

public:
    CiraRuntimeCodeGenerator(llvm::raw_ostream &output) : os(output) {}

    void increaseIndent() {
        indentLevel++;
        indent = std::string(indentLevel * 2, ' ');
    }

    void decreaseIndent() {
        if (indentLevel > 0) {
            indentLevel--;
            indent = std::string(indentLevel * 2, ' ');
        }
    }

    std::string getNewVarName() {
        return "var_" + std::to_string(varCounter++);
    }

    std::string getValueName(Value val) {
        if (valueNames.find(val) == valueNames.end()) {
            valueNames[val] = getNewVarName();
        }
        return valueNames[val];
    }

    // Generate the complete C++ file
    void generateFile(ModuleOp module) {
        // Generate includes
        os << "#include \"CiraRuntime.h\"\n";
        os << "#include <iostream>\n";
        os << "#include <memory>\n";
        os << "#include <cstdint>\n\n";

        os << "using namespace cira::runtime;\n\n";

        // Generate main function
        generateMainFunction(module);
    }

    void generateMainFunction(ModuleOp module) {
        os << "int main(int argc, char** argv) {\n";
        increaseIndent();

        // Initialize runtime
        generateRuntimeInit();

        // Process each function in the module
        module.walk([&](func::FuncOp funcOp) {
            if (funcOp.getName() != "main") {
                generateFunction(funcOp);
            }
        });

        // Cleanup runtime
        generateRuntimeCleanup();

        os << indent << "return 0;\n";
        decreaseIndent();
        os << "}\n\n";
    }

    void generateRuntimeInit() {
        os << indent << "// Initialize Cira runtime\n";
        os << indent << "auto runtime = CiraRuntime::create();\n";
        os << indent << "auto* offload_engine = runtime->getOffloadEngine();\n";
        os << indent << "auto* memory_manager = runtime->getMemoryManager();\n";
        os << indent << "auto* prefetch_controller = runtime->getPrefetchController();\n\n";

        os << indent << "// Configure runtime\n";
        os << indent << "runtime->setVerbosity(1);\n";
        os << indent << "runtime->enableProfiling(true);\n\n";

        runtimeInitialized = true;
    }

    void generateRuntimeCleanup() {
        os << "\n" << indent << "// Cleanup runtime\n";
        os << indent << "// Runtime cleanup is automatic via RAII\n";
    }

    void generateFunction(func::FuncOp funcOp) {
        os << indent << "// Function: " << funcOp.getName() << "\n";
        os << indent << "{\n";
        increaseIndent();

        // Process the function body
        funcOp.walk([&](Operation *op) {
            generateOperation(op);
        });

        decreaseIndent();
        os << indent << "}\n\n";
    }

    void generateOperation(Operation *op) {
        // Handle different operation types
        if (auto loadEdgeOp = dyn_cast<LoadEdgeOp>(op)) {
            generateLoadEdge(loadEdgeOp);
        } else if (auto loadNodeOp = dyn_cast<LoadNodeOp>(op)) {
            generateLoadNode(loadNodeOp);
        } else if (auto getPaddrOp = dyn_cast<GetPaddrOp>(op)) {
            generateGetPaddr(getPaddrOp);
        } else if (auto evictEdgeOp = dyn_cast<EvictEdgeOp>(op)) {
            generateEvictEdge(evictEdgeOp);
        } else if (auto callOp = dyn_cast<CallOp>(op)) {
            generateCall(callOp);
        } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            generateForLoop(forOp);
        }
    }

    void generateLoadEdge(LoadEdgeOp op) {
        std::string resultName = getValueName(op.getResult());
        std::string edgePtr = getValueName(op.getEdgePtr());
        std::string index = getValueName(op.getIndex());

        os << indent << "// Load edge data\n";
        os << indent << "void* " << resultName << " = offload_engine->loadEdge(\n";
        os << indent << "    static_cast<RemoteMemRef*>(" << edgePtr << "),\n";
        os << indent << "    " << index;

        if (op.getPrefetchDistance()) {
            std::string prefetchDist = getValueName(op.getPrefetchDistance());
            os << ",\n" << indent << "    " << prefetchDist;
        } else {
            os << ", 0";
        }

        os << ");\n\n";
    }

    void generateLoadNode(LoadNodeOp op) {
        std::string resultName = getValueName(op.getResult());
        std::string edgeElement = getValueName(op.getEdgeElement());
        std::string fieldName = op.getFieldName().str();

        os << indent << "// Load node data for field: " << fieldName << "\n";
        os << indent << "void* " << resultName << " = offload_engine->loadNode(\n";
        os << indent << "    " << edgeElement << ",\n";
        os << indent << "    \"" << fieldName << "\"";

        if (op.getPrefetchDistance()) {
            std::string prefetchDist = getValueName(op.getPrefetchDistance());
            os << ",\n" << indent << "    " << prefetchDist;
        } else {
            os << ", 0";
        }

        os << ");\n\n";
    }

    void generateGetPaddr(GetPaddrOp op) {
        std::string resultName = getValueName(op.getPaddr());
        std::string nodeData = getValueName(op.getNodeData());
        std::string fieldName = op.getFieldName().str();

        os << indent << "// Get physical address for field: " << fieldName << "\n";
        os << indent << "uintptr_t " << resultName << " = offload_engine->getPhysicalAddr(\n";
        os << indent << "    \"" << fieldName << "\",\n";
        os << indent << "    " << nodeData << ");\n\n";
    }

    void generateEvictEdge(EvictEdgeOp op) {
        std::string edgePtr = getValueName(op.getEdgePtr());
        std::string index = getValueName(op.getIndex());

        os << indent << "// Evict edge from cache\n";
        os << indent << "offload_engine->evictEdge(\n";
        os << indent << "    static_cast<RemoteMemRef*>(" << edgePtr << "),\n";
        os << indent << "    " << index << ");\n\n";
    }

    void generateCall(CallOp op) {
        std::string funcName = op.getCallee().str();

        os << indent << "// Call function: " << funcName << "\n";
        os << indent << funcName << "(";

        bool first = true;
        for (Value operand : op.getOperands()) {
            if (!first) os << ", ";
            os << getValueName(operand);
            first = false;
        }

        os << ");\n\n";
    }

    void generateForLoop(scf::ForOp forOp) {
        std::string loopVar = getValueName(forOp.getInductionVar());
        std::string lowerBound = getValueName(forOp.getLowerBound());
        std::string upperBound = getValueName(forOp.getUpperBound());
        std::string step = getValueName(forOp.getStep());

        os << indent << "// For loop\n";
        os << indent << "for (size_t " << loopVar << " = " << lowerBound << "; ";
        os << loopVar << " < " << upperBound << "; ";
        os << loopVar << " += " << step << ") {\n";

        increaseIndent();

        // Generate loop body
        for (Operation &op : forOp.getBody()->getOperations()) {
            generateOperation(&op);
        }

        decreaseIndent();
        os << indent << "}\n\n";
    }
};

// Compiler backend pass that generates runtime code
class CiraRuntimeBackendPass : public PassWrapper<CiraRuntimeBackendPass, OperationPass<ModuleOp>> {
public:
    StringRef getArgument() const final { return "cira-runtime-backend"; }
    StringRef getDescription() const final {
        return "Generate C++ code using Cira runtime library";
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();

        // Create output file
        std::error_code ec;
        llvm::raw_fd_ostream outputFile("cira_generated.cpp", ec);

        if (ec) {
            module.emitError() << "Failed to create output file: " << ec.message();
            signalPassFailure();
            return;
        }

        // Generate runtime code
        CiraRuntimeCodeGenerator generator(outputFile);
        generator.generateFile(module);

        llvm::outs() << "Generated runtime code to cira_generated.cpp\n";
    }
};

// Example usage: Generate optimized runtime code for graph processing
void generateGraphProcessingCode(ModuleOp module, llvm::raw_ostream &os) {
    os << R"cpp(
#include "CiraRuntime.h"
#include <iostream>
#include <chrono>

using namespace cira::runtime;
using namespace std::chrono;

struct Edge {
    uint64_t from;
    uint64_t to;
    float weight;
};

struct Node {
    float value;
    uint32_t degree;
};

void update_node(Edge* edge, Node* from_node, Node* to_node) {
    // Graph algorithm update logic
    to_node->value += from_node->value * edge->weight;
}

int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_edges> <num_vertices>\n";
        return 1;
    }

    size_t num_edges = std::stoull(argv[1]);
    size_t num_vertices = std::stoull(argv[2]);

    // Initialize runtime
    auto runtime = CiraRuntime::create();
    auto* offload_engine = runtime->getOffloadEngine();
    auto* memory_manager = runtime->getMemoryManager();

    // Configure runtime
    runtime->setVerbosity(1);
    runtime->enableProfiling(true);

    // Allocate graph in remote memory
    auto* edge_data = memory_manager->allocateRemote(
        sizeof(Edge) * num_edges,
        MemoryTier::CXL_ATTACHED
    );

    auto* vertex_data = memory_manager->allocateRemote(
        sizeof(Node) * num_vertices,
        MemoryTier::LOCAL_DRAM
    );

    // Graph traversal parameters
    const size_t cache_line_size = 8;  // 8 edges per cache line
    const size_t prefetch_distance = 2;
    const size_t node_prefetch_distance = 1;

    // Start timing
    auto start = high_resolution_clock::now();

    // Main graph processing loop
    for (size_t i = 0; i < num_edges; i += cache_line_size) {
        // Prefetch next cache line of edges
        offload_engine->loadEdge(edge_data, i, prefetch_distance);

        // Process edges in current cache line
        for (size_t j = 0; j < cache_line_size && (i + j) < num_edges; j++) {
            // Load edge element
            Edge* edge = static_cast<Edge*>(
                offload_engine->loadEdge(edge_data, i + j, 0)
            );

            // Prefetch node data
            Node* from_node = static_cast<Node*>(
                offload_engine->loadNode(edge, "from", node_prefetch_distance)
            );
            Node* to_node = static_cast<Node*>(
                offload_engine->loadNode(edge, "to", node_prefetch_distance)
            );

            // Get physical addresses for direct manipulation
            uintptr_t from_paddr = offload_engine->getPhysicalAddr("from", from_node);
            uintptr_t to_paddr = offload_engine->getPhysicalAddr("to", to_node);

            // Execute update function
            update_node(edge,
                       reinterpret_cast<Node*>(from_paddr),
                       reinterpret_cast<Node*>(to_paddr));
        }

        // Evict processed cache line
        offload_engine->evictEdge(edge_data, i);
    }

    // Stop timing
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    std::cout << "Graph processing completed in " << duration.count() << " ms\n";
    std::cout << "Edges processed: " << num_edges << "\n";
    std::cout << "Vertices: " << num_vertices << "\n";

    // Get memory statistics
    std::cout << "\nMemory usage:\n";
    std::cout << "  CXL memory: "
              << memory_manager->getUsedMemory(MemoryTier::CXL_ATTACHED) / (1024*1024)
              << " MB\n";
    std::cout << "  Local DRAM: "
              << memory_manager->getUsedMemory(MemoryTier::LOCAL_DRAM) / (1024*1024)
              << " MB\n";

    // Cleanup
    memory_manager->deallocateRemote(edge_data);
    memory_manager->deallocateRemote(vertex_data);

    return 0;
}
)cpp";
}

// Registration
std::unique_ptr<Pass> createCiraRuntimeBackendPass() {
    return std::make_unique<CiraRuntimeBackendPass>();
}

void registerCiraRuntimeBackendPass() {
    PassRegistration<CiraRuntimeBackendPass>();
}

} // namespace cira
} // namespace mlir