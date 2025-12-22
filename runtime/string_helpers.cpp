// Provide properly-compiled std::string operations to override broken ClangIR versions
// Compile with: clang++ -c -O2 -fPIC string_helpers.cpp -o string_helpers.o

#include <string>
#include <cstring>

// Force these template instantiations to be compiled properly
namespace std {
    template class __cxx11::basic_string<char, char_traits<char>, allocator<char>>;
}

// Explicit instantiations of commonly-used std::string operations
extern "C" {
    // Default constructor
    void __cxxabiv1_string_default_ctor(std::string* s) {
        new (s) std::string();
    }

    // Destructor
    void __cxxabiv1_string_dtor(std::string* s) {
        s->~basic_string();
    }

    // Constructor from C string
    void __cxxabiv1_string_cstr_ctor(std::string* s, const char* cstr) {
        new (s) std::string(cstr);
    }
}
