#include "cira_shared_arena.h"

#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <new>

#include <sys/mman.h>

struct cira_shared_arena {
    uint8_t *base = nullptr;
    uint64_t size = 0;
    uint64_t control_bytes = 0;
    uint64_t next_offset = 0;
    bool owns_mapping = false;
    std::mutex allocation_mutex;
};

namespace {

std::mutex g_default_mutex;
cira_shared_arena_t *g_default_arena = nullptr;
bool g_default_probed = false;

bool is_power_of_two(uint64_t value) {
    return value && (value & (value - 1)) == 0;
}

bool parse_env_u64(const char *name, uint64_t &out) {
    const char *text = std::getenv(name);
    if (!text || !*text || *text == '-')
        return false;
    char *end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(text, &end, 0);
    if (errno || end == text || *end != '\0')
        return false;
    out = static_cast<uint64_t>(parsed);
    return true;
}

bool env_flag(const char *name) {
    const char *value = std::getenv(name);
    return value && *value && std::strcmp(value, "0") != 0;
}

} // namespace

extern "C" {

int cira_shared_arena_open(cira_shared_arena_t **out, void *base,
                           uint64_t bytes, uint64_t control_bytes) {
    if (!out)
        return CIRA_ARENA_EINVAL;
    *out = nullptr;
    if (!base || bytes == 0 || control_bytes > bytes)
        return CIRA_ARENA_EINVAL;

    const uintptr_t base_address = reinterpret_cast<uintptr_t>(base);
    if (bytes > std::numeric_limits<uintptr_t>::max() - base_address)
        return CIRA_ARENA_ERANGE;

    auto *arena = new (std::nothrow) cira_shared_arena();
    if (!arena)
        return CIRA_ARENA_ENOMEM;
    arena->base = static_cast<uint8_t *>(base);
    arena->size = bytes;
    arena->control_bytes = control_bytes;
    arena->next_offset = control_bytes;
    *out = arena;
    return CIRA_ARENA_OK;
}

void cira_shared_arena_close(cira_shared_arena_t *arena) {
    if (!arena)
        return;
    if (arena->owns_mapping)
        munmap(arena->base, static_cast<size_t>(arena->size));
    delete arena;
}

int cira_shared_arena_alloc(cira_shared_arena_t *arena, uint64_t bytes,
                            uint64_t alignment, uint64_t *out_offset) {
    if (!arena || !out_offset || !is_power_of_two(alignment))
        return CIRA_ARENA_EINVAL;

    std::lock_guard<std::mutex> lock(arena->allocation_mutex);
    const uint64_t mask = alignment - 1;
    if (arena->next_offset > std::numeric_limits<uint64_t>::max() - mask)
        return CIRA_ARENA_ENOMEM;
    const uint64_t aligned = (arena->next_offset + mask) & ~mask;
    if (aligned > arena->size || bytes > arena->size - aligned)
        return CIRA_ARENA_ENOMEM;

    *out_offset = aligned;
    arena->next_offset = aligned + bytes;
    return CIRA_ARENA_OK;
}

int cira_shared_arena_to_offset(const cira_shared_arena_t *arena,
                                const void *ptr, uint64_t bytes,
                                uint64_t *out_offset) {
    if (!arena || !ptr || !out_offset)
        return CIRA_ARENA_EINVAL;

    const uintptr_t base = reinterpret_cast<uintptr_t>(arena->base);
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    if (address < base)
        return CIRA_ARENA_ERANGE;
    const uint64_t offset = static_cast<uint64_t>(address - base);
    if (offset > arena->size || bytes > arena->size - offset)
        return CIRA_ARENA_ERANGE;

    *out_offset = offset;
    return CIRA_ARENA_OK;
}

void *cira_shared_arena_from_offset(const cira_shared_arena_t *arena,
                                    uint64_t offset, uint64_t bytes,
                                    uint64_t alignment) {
    if (!arena || !is_power_of_two(alignment) || (offset & (alignment - 1)))
        return nullptr;
    if (offset > arena->size || bytes > arena->size - offset)
        return nullptr;
    return arena->base + offset;
}

cira_shared_arena_t *cira_shared_arena_default(void) {
    std::lock_guard<std::mutex> lock(g_default_mutex);
    if (g_default_arena || g_default_probed)
        return g_default_arena;
    g_default_probed = true;

    uint64_t size = 0;
    if (!parse_env_u64("CIRA_CXL_ARENA_SIZE", size) || size == 0)
        return nullptr;
    uint64_t control_bytes = CIRA_CXL_CONTROL_BYTES;
    if (std::getenv("CIRA_CXL_ARENA_CONTROL_BYTES") &&
        !parse_env_u64("CIRA_CXL_ARENA_CONTROL_BYTES", control_bytes))
        return nullptr;

    uint64_t address = 0;
    const bool has_address = parse_env_u64("CIRA_CXL_ARENA_ADDR", address) && address;
    const bool strict = env_flag("CIRA_CXL_ARENA_STRICT") || env_flag("CIRA_CXL_MMIO_STRICT");
    if (!has_address && strict)
        return nullptr;

    if (has_address && address > std::numeric_limits<uintptr_t>::max())
        return nullptr;
    void *base = reinterpret_cast<void *>(static_cast<uintptr_t>(address));
    bool owns_mapping = false;
    if (!has_address) {
        if (size > std::numeric_limits<size_t>::max())
            return nullptr;
        base = mmap(nullptr, static_cast<size_t>(size), PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED)
            return nullptr;
        owns_mapping = true;
    }

    cira_shared_arena_t *arena = nullptr;
    if (cira_shared_arena_open(&arena, base, size, control_bytes) != CIRA_ARENA_OK) {
        if (owns_mapping)
            munmap(base, static_cast<size_t>(size));
        return nullptr;
    }
    arena->owns_mapping = owns_mapping;
    g_default_arena = arena;
    return g_default_arena;
}

} // extern "C"
