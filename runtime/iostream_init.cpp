// Force iostream initialization before any other static constructors
// This is needed because ClangIR doesn't generate the std::ios_base::Init
// static object initialization
#include <iostream>

namespace {
    // Force construction of iostream objects at static init time
    // with high priority (runs before other constructors)
    struct __iostream_force_init {
        __iostream_force_init() {
            // Access cout/cin/cerr to force their initialization
            (void)std::cout.rdbuf();
            (void)std::cin.rdbuf();
            (void)std::cerr.rdbuf();
        }
    };

    // Use constructor priority to run this very early
    __attribute__((init_priority(101)))
    __iostream_force_init __iostream_init_instance;
}
