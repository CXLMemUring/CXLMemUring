// Native support object for the ClangIR/CIRA GAPBS build.
//
// The CIR path can lower the inline command-line class methods but miss the
// weak C++ vtables for these header-only polymorphic classes. Keeping this
// small TU on the native C++ path gives each GAPBS binary the standard ABI
// support symbols without changing benchmark behavior.

#include "command_line.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

asm(".globl _ZSt19piecewise_construct\n"
    ".type _ZSt19piecewise_construct,@object\n"
    ".size _ZSt19piecewise_construct,1\n"
    "_ZSt19piecewise_construct:\n"
    ".byte 0\n");

extern "C" __attribute__((used)) void gapbs_force_command_line_vtables(
    int argc, char **argv) {
  CLBase base(argc, argv, "base");
  CLApp app(argc, argv, "app");
  CLIterApp iter(argc, argv, "iter", 1);
  CLPageRank pagerank(argc, argv, "pagerank", 1e-4, 20);
  CLDelta<int> delta_i(argc, argv, "delta_i");
  CLDelta<float> delta_f(argc, argv, "delta_f");
  CLConvert convert(argc, argv, "convert");
}

#ifdef CIRA_GAPBS_STATIC_MARKER
namespace {

constexpr uint64_t kCiraCsrPrefetchRecords = 1ULL << 0;
constexpr uint64_t kCiraCsrRecordSpan = 1ULL << 3;

struct GapbsCsrGraphHeader {
  bool directed;
  char padding[7];
  int64_t num_nodes;
  int64_t num_edges;
  const void* const* out_index;
  const void* out_neighbors;
  const void* const* in_index;
  const void* in_neighbors;
};

static_assert(offsetof(GapbsCsrGraphHeader, num_nodes) == 8,
              "Unexpected GAPBS CSRGraph layout");
static_assert(offsetof(GapbsCsrGraphHeader, out_index) == 24,
              "Unexpected GAPBS CSRGraph layout");
static_assert(sizeof(GapbsCsrGraphHeader) == 56,
              "Unexpected GAPBS CSRGraph layout");

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value && *value && std::strcmp(value, "0") != 0;
}

bool parse_env_u64(const char* name, uint64_t& value) {
  const char* text = std::getenv(name);
  if (!text || !*text)
    return false;

  char* end = nullptr;
  unsigned long long parsed = std::strtoull(text, &end, 0);
  if (end == text || (end && *end != '\0'))
    return false;

  value = static_cast<uint64_t>(parsed);
  return true;
}

#if defined(__x86_64__)
uint64_t gem5_m5_cira_prefetch(const void* addr, uint64_t size) {
  uint64_t ret = 0;
  asm volatile(
      ".byte 0x0F, 0x04\n\t"
      ".word 0x5c"
      : "=a"(ret)
      : "D"(reinterpret_cast<uint64_t>(addr)), "S"(size)
      : "memory");
  return ret;
}

uint64_t gem5_m5_cira_prefetch_csr(uint64_t offsets_addr,
                                   uint64_t records_addr,
                                   uint64_t values_addr,
                                   uint64_t row_start,
                                   uint64_t row_count,
                                   uint64_t packed) {
  uint64_t ret = 0;
  register uint64_t r8 asm("r8") = row_count;
  register uint64_t r9 asm("r9") = packed;
  asm volatile(
      ".byte 0x0F, 0x04\n\t"
      ".word 0x61"
      : "=a"(ret)
      : "D"(offsets_addr), "S"(records_addr), "d"(values_addr),
        "c"(row_start), "r"(r8), "r"(r9)
      : "memory");
  return ret;
}
#else
uint64_t gem5_m5_cira_prefetch(const void*, uint64_t) {
  return 0;
}

uint64_t gem5_m5_cira_prefetch_csr(uint64_t, uint64_t, uint64_t,
                                   uint64_t, uint64_t, uint64_t) {
  return 0;
}
#endif

uint64_t gapbs_record_stride(uint32_t benchmark_id) {
  uint64_t stride = benchmark_id == 1 ? 8 : 4;
  parse_env_u64("CIRA_GAPBS_RECORD_STRIDE", stride);
  return stride;
}

bool emit_gapbs_gem5_cira_marker(uint32_t benchmark_id, const void* addr,
                                 uint64_t bytes) {
  if (!addr)
    return false;

  auto* graph = static_cast<const GapbsCsrGraphHeader*>(addr);
  const uint64_t record_stride = gapbs_record_stride(benchmark_id);
  if (record_stride == 0 || record_stride > 0xffff ||
      graph->num_nodes <= 0 || !graph->out_index) {
    return gem5_m5_cira_prefetch(addr, bytes ? bytes : 64) != 0;
  }

  const uint64_t node_count = static_cast<uint64_t>(graph->num_nodes);
  uint64_t row_start = 0;
  parse_env_u64("CIRA_GAPBS_CSR_ROW_START", row_start);
  if (row_start > node_count)
    row_start = node_count;

  uint64_t row_count = node_count - row_start;
  uint64_t row_limit = 0;
  if (parse_env_u64("CIRA_GAPBS_CSR_ROW_LIMIT", row_limit) &&
      row_limit < row_count) {
    row_count = row_limit;
  }
  if (row_count == 0)
    return false;

  const uint64_t row_end = row_start + row_count;
  const auto record_begin =
      reinterpret_cast<uint64_t>(graph->out_index[row_start]);
  const auto record_end = reinterpret_cast<uint64_t>(graph->out_index[row_end]);
  if (record_begin == 0 || record_end <= record_begin)
    return gem5_m5_cira_prefetch(addr, bytes ? bytes : 64) != 0;

  const uint64_t flags = kCiraCsrPrefetchRecords | kCiraCsrRecordSpan;
  const uint64_t packed = ((record_stride & 0xffffULL) << 8) |
                          ((flags & 0xffULL) << 56);
  return gem5_m5_cira_prefetch_csr(record_begin, record_end, 0, row_start,
                                   row_count, packed) != 0;
}

} // namespace

extern "C" __attribute__((used, noinline)) void cira_gapbs_region_marker(
    uint32_t benchmark_id, uint32_t region_id, const void* addr,
    uint64_t bytes) {
  (void)region_id;
  if (env_flag_enabled("CIRA_GAPBS_MARKER_ONLY"))
    return;

  if (env_flag_enabled("CIRA_GEM5_M5OPS") ||
      env_flag_enabled("CIRA_GAPBS_GEM5_M5OPS")) {
    (void)emit_gapbs_gem5_cira_marker(benchmark_id, addr, bytes);
  }
}
#endif
