OUTPUT_ARCH(riscv)
ENTRY(_start)

PHDRS
{
    text PT_LOAD FLAGS(5);
    data PT_LOAD FLAGS(6);
}

MEMORY
{
    ram (rwx) : ORIGIN = 0x00010000, LENGTH = 16M
}

SECTIONS
{
    . = ORIGIN(ram);
    .text : ALIGN(16) {
        KEEP(*(.text.start))
        *(.text .text.*)
    } > ram :text
    .rodata : ALIGN(16) { *(.rodata .rodata.*) *(.srodata .srodata.*) } > ram :text
    .data : ALIGN(16) {
        __global_pointer$ = . + 0x800;
        *(.sdata .sdata.*)
        *(.data .data.*)
    } > ram :data
    .bss (NOLOAD) : ALIGN(16) {
        __bss_start = .;
        *(.sbss .sbss.*)
        *(.bss .bss.* COMMON)
        __bss_end = .;
    } > ram :data
    .stack (NOLOAD) : ALIGN(16) {
        __cira_stack_bottom = .;
        . += 64K;
        __cira_stack_top = .;
    } > ram :data
    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
