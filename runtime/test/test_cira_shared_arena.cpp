#include "cira_cxl_job.h"
#include "cira_shared_arena.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr uint64_t kArenaBytes = 0x4000;

void test_allocation_and_translation() {
    alignas(CIRA_CXL_CACHELINE_SIZE) std::byte storage[kArenaBytes]{};
    cira_shared_arena_t *arena = reinterpret_cast<cira_shared_arena_t *>(1);
    assert(cira_shared_arena_open(&arena, nullptr, sizeof(storage),
                                  CIRA_CXL_CONTROL_BYTES) == CIRA_ARENA_EINVAL);
    assert(arena == nullptr);
    assert(cira_shared_arena_open(&arena, storage, sizeof(storage),
                                  CIRA_CXL_CONTROL_BYTES) == CIRA_ARENA_OK);

    uint64_t completion_offset = 0;
    assert(cira_shared_arena_alloc(arena, CIRA_CXL_CACHELINE_SIZE,
                                   CIRA_CXL_CACHELINE_SIZE,
                                   &completion_offset) == CIRA_ARENA_OK);
    assert(completion_offset == CIRA_CXL_CONTROL_BYTES);
    assert((completion_offset & (CIRA_CXL_CACHELINE_SIZE - 1)) == 0);

    void *completion = cira_shared_arena_from_offset(
        arena, completion_offset, CIRA_CXL_CACHELINE_SIZE,
        CIRA_CXL_CACHELINE_SIZE);
    assert(completion == storage + completion_offset);

    uint64_t round_trip = 0;
    assert(cira_shared_arena_to_offset(arena, completion,
                                       CIRA_CXL_CACHELINE_SIZE,
                                       &round_trip) == CIRA_ARENA_OK);
    assert(round_trip == completion_offset);
    cira_shared_arena_close(arena);
}

void test_exact_end_and_bounds() {
    alignas(CIRA_CXL_CACHELINE_SIZE) std::byte storage[0x3000]{};
    cira_shared_arena_t *arena = nullptr;
    assert(cira_shared_arena_open(&arena, storage, sizeof(storage),
                                  CIRA_CXL_CONTROL_BYTES) == CIRA_ARENA_OK);

    uint64_t offset = 0;
    assert(cira_shared_arena_alloc(arena, 0x1000, 1, &offset) == CIRA_ARENA_OK);
    assert(offset == CIRA_CXL_CONTROL_BYTES);
    assert(cira_shared_arena_from_offset(arena, offset, 0x1000, 1) ==
           storage + offset);
    assert(cira_shared_arena_from_offset(arena, sizeof(storage), 0, 1) ==
           storage + sizeof(storage));
    assert(cira_shared_arena_to_offset(arena, storage + sizeof(storage), 0,
                                       &offset) == CIRA_ARENA_OK);
    assert(offset == sizeof(storage));
    assert(cira_shared_arena_from_offset(arena, sizeof(storage), 1, 1) == nullptr);
    assert(cira_shared_arena_from_offset(
               arena, std::numeric_limits<uint64_t>::max(), 2, 1) == nullptr);
    assert(cira_shared_arena_alloc(arena, 1, 1, &offset) == CIRA_ARENA_ENOMEM);

    std::byte foreign[8]{};
    assert(cira_shared_arena_to_offset(arena, foreign, sizeof(foreign),
                                       &offset) == CIRA_ARENA_ERANGE);
    assert(cira_shared_arena_to_offset(arena, storage + sizeof(storage), 1,
                                       &offset) == CIRA_ARENA_ERANGE);
    assert(cira_shared_arena_alloc(arena, 1, 3, &offset) == CIRA_ARENA_EINVAL);
    cira_shared_arena_close(arena);
}

void test_strict_default_requires_mapping() {
    pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        unsetenv("CIRA_CXL_ARENA_ADDR");
        unsetenv("CIRA_CXL_ARENA_SIZE");
        setenv("CIRA_CXL_ARENA_STRICT", "1", 1);
        _exit(cira_shared_arena_default() == nullptr ? 0 : 1);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

void test_explicit_default_mapping() {
    alignas(CIRA_CXL_CACHELINE_SIZE) static std::byte storage[kArenaBytes]{};
    char address[32]{};
    char size[32]{};
    std::snprintf(address, sizeof(address), "%llu",
                  static_cast<unsigned long long>(
                      reinterpret_cast<uintptr_t>(storage)));
    std::snprintf(size, sizeof(size), "%llu",
                  static_cast<unsigned long long>(sizeof(storage)));
    setenv("CIRA_CXL_ARENA_ADDR", address, 1);
    setenv("CIRA_CXL_ARENA_SIZE", size, 1);
    setenv("CIRA_CXL_ARENA_CONTROL_BYTES", "8192", 1);
    setenv("CIRA_CXL_ARENA_STRICT", "1", 1);

    cira_shared_arena_t *arena = cira_shared_arena_default();
    assert(arena != nullptr);
    assert(cira_shared_arena_from_offset(arena, CIRA_CXL_CONTROL_BYTES, 1, 1) ==
           storage + CIRA_CXL_CONTROL_BYTES);
}

} // namespace

int main() {
    test_allocation_and_translation();
    test_exact_end_and_bounds();
    test_strict_default_requires_mapping();
    test_explicit_default_mapping();
    return 0;
}
