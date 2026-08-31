    .section .text.start,"ax",@progbits
    .globl _start
    .type _start,@function
_start:
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    la sp, __cira_stack_top
    call cira_device_main
1:
    j 1b
    .size _start, .-_start
