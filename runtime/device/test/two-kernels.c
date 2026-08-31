#include "cira_rv64_device.h"

void __cira_rv64_0000000000000002(uint64_t arena_base, uint64_t args_offset) {
    (void)arena_base;
    cira_device_install_cacheline(args_offset);
}

void __cira_rv64_0000000000000009(uint64_t arena_base, uint64_t args_offset) {
    (void)arena_base;
    uint64_t offset = cira_device_checked_load(args_offset, sizeof(uint64_t), _Alignof(uint64_t));
    cira_device_install_cacheline(offset);
}
