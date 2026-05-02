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

    cira_future_free(completion);
    cira_llc_tile_free(auto_tile);
    cira_llc_tile_free(tile);
    return 0;
}
