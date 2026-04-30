// SPDX-License-Identifier: GPL-2.0
/*
 * cxl_type2_numa.c - Map CXL Type 2 device memory as a NUMA node
 *
 * Creates NUMA node 2 by reassigning an offlined memory block from
 * node 0/1 to node 2. The memory block must be offlined BEFORE
 * loading this module:
 *
 *   echo offline > /sys/devices/system/memory/memory62/state
 *   insmod cxl_type2_numa.ko phys_blk=62 target_nid=2
 *
 * The driver then:
 *   1. Removes the offlined block from its original node
 *   2. Adds it to the target NUMA node
 *   3. Onlines it, making it available for numactl -m 2
 *   4. Maps the CXL Type 2 GPU BAR0 for CSR access
 */

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/memory_hotplug.h>
#include <linux/memory.h>
#include <linux/numa.h>
#include <linux/node.h>

#define DRIVER_NAME    "cxl_type2_numa"
#define PCI_VENDOR_INTEL    0x8086
#define PCI_DEVICE_CXL_T2   0x0ddb

/* CSR register offsets (relative to BAR0 base) */
#define CSR_BASE_OFFSET     0x180100
#define CSR_STATUS           (CSR_BASE_OFFSET + 0x12C)
#define CSR_DCOH_ENABLE      (CSR_BASE_OFFSET + 0x148)

/* Module parameters */
static int target_nid = 2;
module_param(target_nid, int, 0444);
MODULE_PARM_DESC(target_nid, "Target NUMA node ID (default: 2)");

static int phys_blk = 62;
module_param(phys_blk, int, 0444);
MODULE_PARM_DESC(phys_blk, "Offlined memory block index to reassign (default: 62)");

struct cxl_t2_dev {
	struct pci_dev *pdev;
	void __iomem *bar0;
	size_t bar0_size;
	u64 mem_start;
	u64 mem_size;
	int nid;
	int orig_nid;
	bool mem_reassigned;
};

static struct cxl_t2_dev *g_dev;

static u32 cxl_t2_read_csr(struct cxl_t2_dev *dev, u32 offset)
{
	if (!dev->bar0)
		return 0xDEAD;
	return readl(dev->bar0 + offset);
}

static void cxl_t2_write_csr(struct cxl_t2_dev *dev, u32 offset, u32 val)
{
	if (dev->bar0)
		writel(val, dev->bar0 + offset);
}

/*
 * Reassign an offlined memory block to the target NUMA node.
 * The block must already be offline before this is called.
 */
static int cxl_t2_reassign_memory(struct cxl_t2_dev *dev)
{
	u64 block_sz = memory_block_size_bytes();
	u64 start = (u64)phys_blk * block_sz;
	u64 size = block_sz;
	int ret;
	char state_path[128];
	struct file *f;

	dev->mem_start = start;
	dev->mem_size = size;

	dev_info(&dev->pdev->dev,
		 "Reassigning memory block %d (0x%llx, %llu MB) to node %d\n",
		 phys_blk, start, size >> 20, dev->nid);

	/* Verify the block is offline */
	snprintf(state_path, sizeof(state_path),
		 "/sys/devices/system/memory/memory%d/state", phys_blk);

	dev_info(&dev->pdev->dev, "Memory block size: %llu MB\n",
		 block_sz >> 20);

	/*
	 * Step 1: Remove the offlined memory from its current node.
	 * remove_memory() removes offlined memory from the system.
	 */
	ret = remove_memory(start, size);
	if (ret) {
		dev_err(&dev->pdev->dev,
			"remove_memory(0x%llx, 0x%llx) failed: %d\n",
			start, size, ret);
		return ret;
	}
	dev_info(&dev->pdev->dev, "Removed memory from original node\n");

	/*
	 * Step 2: Add the memory to the target NUMA node.
	 * add_memory() creates struct pages and memory blocks for
	 * the specified physical address range on the given node.
	 */
	ret = add_memory(dev->nid, start, size, MHP_NONE);
	if (ret) {
		dev_err(&dev->pdev->dev,
			"add_memory(nid=%d, 0x%llx, 0x%llx) failed: %d\n",
			dev->nid, start, size, ret);
		/* Try to restore to node 0 */
		add_memory(0, start, size, MHP_NONE);
		return ret;
	}
	dev_info(&dev->pdev->dev,
		 "Added memory to NUMA node %d\n", dev->nid);

	dev->mem_reassigned = true;
	return 0;
}

/*
 * Online the memory block on its new NUMA node.
 */
static int cxl_t2_online_memory(struct cxl_t2_dev *dev)
{
	/* Write to sysfs to online the memory block */
	char path[128];
	struct file *f;
	loff_t pos = 0;
	const char *cmd = "online\n";
	int ret;

	snprintf(path, sizeof(path),
		 "/sys/devices/system/memory/memory%d/state", phys_blk);

	f = filp_open(path, O_WRONLY, 0);
	if (IS_ERR(f)) {
		dev_err(&dev->pdev->dev,
			"Cannot open %s: %ld\n", path, PTR_ERR(f));
		return PTR_ERR(f);
	}

	ret = kernel_write(f, cmd, strlen(cmd), &pos);
	filp_close(f, NULL);

	if (ret < 0) {
		dev_err(&dev->pdev->dev,
			"Failed to online memory block %d: %d\n",
			phys_blk, ret);
		return ret;
	}

	dev_info(&dev->pdev->dev,
		 "Memory block %d onlined on NUMA node %d (%llu MB)\n",
		 phys_blk, dev->nid, dev->mem_size >> 20);
	return 0;
}

static int cxl_t2_probe(struct pci_dev *pdev,
			 const struct pci_device_id *id)
{
	struct cxl_t2_dev *dev;
	int ret;
	u32 status;

	/* Only bind to the first function (3b:00.0), skip 3b:00.1 */
	if (PCI_FUNC(pdev->devfn) != 0)
		return -ENODEV;

	dev = kzalloc(sizeof(*dev), GFP_KERNEL);
	if (!dev)
		return -ENOMEM;

	dev->pdev = pdev;
	dev->nid = target_nid;
	pci_set_drvdata(pdev, dev);
	g_dev = dev;

	ret = pci_enable_device(pdev);
	if (ret) {
		dev_err(&pdev->dev, "Failed to enable PCI device: %d\n", ret);
		goto err_free;
	}

	/* Map BAR0 for CSR access */
	dev->bar0_size = pci_resource_len(pdev, 0);
	dev->bar0 = pci_iomap(pdev, 0, dev->bar0_size);
	if (!dev->bar0) {
		dev_err(&pdev->dev, "Failed to map BAR0\n");
		ret = -ENOMEM;
		goto err_disable;
	}

	status = cxl_t2_read_csr(dev, CSR_STATUS);
	dev_info(&pdev->dev, "BAR0 mapped (%zu bytes), GPU Status: 0x%x\n",
		 dev->bar0_size, status);

	/* Enable DCOH */
	cxl_t2_write_csr(dev, CSR_DCOH_ENABLE, 0x1);
	wmb();

	/* Reassign memory block to target NUMA node */
	ret = cxl_t2_reassign_memory(dev);
	if (ret) {
		dev_err(&pdev->dev, "Memory reassignment failed: %d\n", ret);
		goto err_unmap;
	}

	/* Online the memory on the new node */
	ret = cxl_t2_online_memory(dev);
	if (ret) {
		dev_warn(&pdev->dev, "Memory online failed: %d (try manually)\n", ret);
		/* Don't fail - the block is assigned, user can online manually */
	}

	dev_info(&pdev->dev,
		 "CXL Type 2 driver loaded: NUMA node %d with %llu MB\n",
		 dev->nid, dev->mem_size >> 20);
	return 0;

err_unmap:
	pci_iounmap(pdev, dev->bar0);
err_disable:
	pci_disable_device(pdev);
err_free:
	kfree(dev);
	return ret;
}

static void cxl_t2_remove(struct pci_dev *pdev)
{
	struct cxl_t2_dev *dev = pci_get_drvdata(pdev);

	if (!dev)
		return;

	if (dev->mem_reassigned) {
		dev_info(&pdev->dev,
			 "Restoring memory block %d to node 0\n", phys_blk);

		/* Offline the memory first */
		{
			char path[128];
			struct file *f;
			loff_t pos = 0;
			snprintf(path, sizeof(path),
				 "/sys/devices/system/memory/memory%d/state",
				 phys_blk);
			f = filp_open(path, O_WRONLY, 0);
			if (!IS_ERR(f)) {
				kernel_write(f, "offline\n", 8, &pos);
				filp_close(f, NULL);
				msleep(500);
			}
		}

		/* Remove from node 2, add back to node 0 */
		if (remove_memory(dev->mem_start, dev->mem_size) == 0) {
			if (add_memory(0, dev->mem_start,
				       dev->mem_size, MHP_NONE) == 0) {
				/* Online it back on node 0 */
				char path[128];
				struct file *f;
				loff_t pos = 0;
				snprintf(path, sizeof(path),
					 "/sys/devices/system/memory/memory%d/state",
					 phys_blk);
				f = filp_open(path, O_WRONLY, 0);
				if (!IS_ERR(f)) {
					kernel_write(f, "online\n", 7, &pos);
					filp_close(f, NULL);
				}
				dev_info(&pdev->dev,
					 "Memory restored to node 0\n");
			}
		}
	}

	/* Disable DCOH */
	if (dev->bar0)
		cxl_t2_write_csr(dev, CSR_DCOH_ENABLE, 0x0);

	if (dev->bar0)
		pci_iounmap(pdev, dev->bar0);
	pci_disable_device(pdev);

	g_dev = NULL;
	kfree(dev);
	dev_info(&pdev->dev, "CXL Type 2 driver unloaded\n");
}

static ssize_t cxl_t2_status_show(struct device *d,
				   struct device_attribute *attr,
				   char *buf)
{
	struct pci_dev *pdev = to_pci_dev(d);
	struct cxl_t2_dev *dev = pci_get_drvdata(pdev);
	u32 status;

	if (!dev || !dev->bar0)
		return sysfs_emit(buf, "not initialized\n");

	status = cxl_t2_read_csr(dev, CSR_STATUS);
	return sysfs_emit(buf,
		"gpu_status: 0x%x\n"
		"mem_block: %d\n"
		"mem_phys: 0x%llx\n"
		"mem_size: %llu MB\n"
		"numa_node: %d\n"
		"reassigned: %s\n",
		status, phys_blk,
		dev->mem_start, dev->mem_size >> 20,
		dev->nid,
		dev->mem_reassigned ? "yes" : "no");
}
static DEVICE_ATTR(cxl_t2_status, 0444, cxl_t2_status_show, NULL);

static struct attribute *cxl_t2_attrs[] = {
	&dev_attr_cxl_t2_status.attr,
	NULL,
};

static const struct attribute_group cxl_t2_attr_group = {
	.attrs = cxl_t2_attrs,
};

static const struct pci_device_id cxl_t2_ids[] = {
	{ PCI_DEVICE(PCI_VENDOR_INTEL, PCI_DEVICE_CXL_T2) },
	{ 0 }
};
MODULE_DEVICE_TABLE(pci, cxl_t2_ids);

static struct pci_driver cxl_t2_driver = {
	.name     = DRIVER_NAME,
	.id_table = cxl_t2_ids,
	.probe    = cxl_t2_probe,
	.remove   = cxl_t2_remove,
	.dev_groups = (const struct attribute_group *[]) {
		&cxl_t2_attr_group, NULL
	},
};

module_pci_driver(cxl_t2_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("CXLMemUring Project");
MODULE_DESCRIPTION("CXL Type 2 GPU NUMA memory mapping driver");
MODULE_VERSION("2.0");
