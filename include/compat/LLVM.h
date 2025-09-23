#ifndef CIRA_COMPAT_LLVM_H
#define CIRA_COMPAT_LLVM_H

#include <algorithm>
#include <memory>
#include <functional>
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"

namespace llvm {
// For ArrayRef<T>
template <typename T>
T* uninitialized_copy(ArrayRef<T> src, T* dest) {
    return std::uninitialized_copy(src.begin(), src.end(), dest);
}

// For StringRef
inline char* uninitialized_copy(StringRef str, char* dest) {
    return std::uninitialized_copy(str.begin(), str.end(), dest);
}

// Generic iterators
template <typename InputIt, typename OutputIt>
OutputIt uninitialized_copy(InputIt first, InputIt last, OutputIt d_first) {
    return std::uninitialized_copy(first, last, d_first);
}

// With size
template <typename InputIt, typename Size, typename OutputIt>
OutputIt uninitialized_copy_n(InputIt first, Size n, OutputIt d_first) {
    return std::uninitialized_copy_n(first, n, d_first);
}

// hash_combine_range for various MLIR range types
template <typename RangeT>
hash_code hash_combine_range(const RangeT& range) {
    hash_code result = hash_value(0);
    for (const auto& elem : range) {
        result = hash_combine(result, hash_value(elem));
    }
    return result;
}

} // namespace llvm

// Additional compatibility for MLIR types
namespace mlir {
class SuccessorRange;
class TypeRange;

// Forward declare hash functions for MLIR types
inline ::llvm::hash_code hash_value(const SuccessorRange& range) {
    // Simple hash implementation - can be improved
    return ::llvm::hash_value(reinterpret_cast<uintptr_t>(&range));
}

inline ::llvm::hash_code hash_value(const TypeRange& range) {
    // Simple hash implementation - can be improved
    return ::llvm::hash_value(reinterpret_cast<uintptr_t>(&range));
}
} // namespace mlir

#endif // CIRA_COMPAT_LLVM_H