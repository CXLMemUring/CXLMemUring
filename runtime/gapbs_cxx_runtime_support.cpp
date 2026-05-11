// Native support object for the ClangIR/CIRA GAPBS build.
//
// The CIR path can lower the inline command-line class methods but miss the
// weak C++ vtables for these header-only polymorphic classes. Keeping this
// small TU on the native C++ path gives each GAPBS binary the standard ABI
// support symbols without changing benchmark behavior.

#include "command_line.h"

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
