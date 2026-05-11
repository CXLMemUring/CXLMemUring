#include "CiraRuntime.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace cira::runtime;

namespace {

int fail(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    return 1;
}

} // namespace

int main() {
    alignas(64) uint8_t source[128];
    for (size_t i = 0; i < sizeof(source); ++i)
        source[i] = static_cast<uint8_t>(i ^ 0x5a);

    alignas(64) uint8_t linear_region[256] = {};
    uint64_t device_base = 0x8000000000ULL;
    if (cira_register_linear_region(linear_region, sizeof(linear_region),
                                    device_base, 0) != 0)
        return fail("failed to register linear CIRA region");
    void* native_next = linear_region + 96;
    uintptr_t translated = cira_translate_paddr("next", native_next);
    if (translated != device_base + 96)
        return fail("registered native pointer did not translate by region offset");
    if (cira_translate_registered_addr(native_next) != device_base + 96)
        return fail("registered-address fast path returned wrong device address");
    if (cira_unregister_linear_region(linear_region) != 0)
        return fail("failed to unregister linear CIRA region");

    void* tile = cira_llc_tile_alloc(sizeof(source));
    if (!tile) return fail("cira_llc_tile_alloc returned null");
    if ((reinterpret_cast<uintptr_t>(tile) & 63) != 0)
        return fail("tile is not cacheline-aligned");
    if (!cira_llc_tile_future(tile))
        return fail("tile has no internal future");
    if (cira_llc_tile_get_mwait(tile) != tile)
        return fail("get_mwait did not return the tile pointer");

    void* installed = cira_llc_tile_install_from_cxl(
        tile, source, sizeof(source), nullptr);
    if (installed != tile)
        return fail("install_from_cxl did not reuse the supplied tile");
    if (cira_llc_tile_get_mwait(tile) != tile)
        return fail("get_mwait failed after install_from_cxl");
    if (std::memcmp(tile, source, sizeof(source)) != 0)
        return fail("tile bytes do not match source bytes");

    void* completion = cira_future_alloc();
    if (!completion) return fail("cira_future_alloc returned null");

    void* auto_tile = cira_llc_tile_install_from_cxl(
        nullptr, source, 64, completion);
    if (!auto_tile) return fail("install_from_cxl did not allocate a tile");
    if (cira_future_await(completion) == nullptr)
        return fail("external completion did not become ready");
    if (std::memcmp(auto_tile, source, 64) != 0)
        return fail("auto-allocated tile bytes do not match source bytes");

    void* pool = cira_future_pool_alloc(4);
    if (!pool) return fail("cira_future_pool_alloc returned null");
    if (cira_future_pool_depth(pool) != 4)
        return fail("future pool reported wrong depth");
    void* pool_entry0 = cira_future_pool_get(pool, 0);
    void* pool_entry2 = cira_future_pool_get(pool, 2);
    if (!pool_entry0 || !pool_entry2)
        return fail("future pool returned null entry");
    if ((reinterpret_cast<uintptr_t>(pool_entry0) & 63) != 0)
        return fail("future pool entry is not cacheline-aligned");
    if (static_cast<uint8_t*>(pool_entry2) -
            static_cast<uint8_t*>(pool_entry0) != 128)
        return fail("future pool entries are not 64-byte strided");

    uint64_t completion_device_base = 0x9000000000ULL;
    if (cira_future_pool_register(pool, completion_device_base, 0) != 0)
        return fail("failed to register future pool");
    if (cira_future_pool_get_device_addr(pool, 2) != completion_device_base + 128)
        return fail("future pool device address did not use registered offset");
    if (cira_future_pool_arm(pool, 2) != 0)
        return fail("failed to arm future pool entry");
    void* pool_tile = cira_llc_tile_install_from_cxl(
        nullptr, source, 64, pool_entry2);
    if (!pool_tile) return fail("pool completion install failed");
    if (cira_future_await(pool_entry2) == nullptr)
        return fail("pool completion did not become ready");

    cira_future_free(completion);
    cira_llc_tile_free(pool_tile);
    cira_future_pool_free(pool);
    cira_llc_tile_free(auto_tile);
    cira_llc_tile_free(tile);
    return 0;
}
