#include <linux/module.h>
#include <linux/io.h>
#include <linux/delay.h>

static unsigned long rcrb_base = 0xa2bbe000;
module_param(rcrb_base, ulong, 0444);

static int __init rcrb_write_init(void)
{
    void __iomem *rcrb;
    u32 fbctl, fbsta, ep_size_lo;

    rcrb = ioremap(rcrb_base, 0x2000);
    if (!rcrb) {
        pr_err("rcrb_write: failed to map RCRB at 0x%lx\n", rcrb_base);
        return -ENOMEM;
    }

    /* Read current Flex Bus Port Control/Status at RCRB+0xEB4/0xEB8 */
    fbctl = readl(rcrb + 0xEB4);
    fbsta = readl(rcrb + 0xEB8);
    pr_info("rcrb_write: Before: FBCtl=0x%08x FBSta=0x%08x\n", fbctl, fbsta);

    /* Enable Cache + IO + Mem (bits 0,1,2) */
    writel(fbctl | 0x07, rcrb + 0xEB4);
    wmb();
    msleep(2000);

    fbctl = readl(rcrb + 0xEB4);
    fbsta = readl(rcrb + 0xEB8);
    pr_info("rcrb_write: After:  FBCtl=0x%08x FBSta=0x%08x\n", fbctl, fbsta);
    pr_info("rcrb_write: Mem Active=%d Cache Active=%d IO Active=%d\n",
            (fbsta >> 2) & 1, fbsta & 1, (fbsta >> 1) & 1);

    iounmap(rcrb);
    return -EAGAIN; /* Unload immediately after doing the write */
}

static void __exit rcrb_write_exit(void) {}

module_init(rcrb_write_init);
module_exit(rcrb_write_exit);
MODULE_LICENSE("GPL");
