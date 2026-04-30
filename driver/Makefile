obj-m += cxl_type2_numa.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install: all
	sudo insmod cxl_type2_numa.ko target_nid=2 cxl_size_mb=2048

uninstall:
	sudo rmmod cxl_type2_numa 2>/dev/null || true

reload: uninstall install

status:
	@echo "=== Module status ==="
	@lsmod | grep cxl_type2 || echo "not loaded"
	@echo ""
	@echo "=== dmesg ==="
	@dmesg | grep -i "cxl_type2\|cxl type 2\|Type2" | tail -20
	@echo ""
	@echo "=== NUMA topology ==="
	@numactl --hardware 2>/dev/null | head -12
	@echo ""
	@echo "=== sysfs ==="
	@cat /sys/bus/pci/devices/0000:3b:00.0/cxl_t2_status 2>/dev/null || echo "no sysfs"

.PHONY: all clean install uninstall reload status
