## Default Config Changes
1. struct-abi = false
2. memref-abi = false
3. c-style-memref = false

## Cgeist Bugfix
1. Default to clang-llvm type converter
2. Disable anonymize functionality
3. [Check reference consistency before store](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L266-L273). 
4. [Add missing initializer logic in new expression](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L1712). With minor changes to some utilities in this function (e.g. memory size calculation, etc.).
5. [Fix pointer type in binary sub when converted to memref types](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L3151-L3169)
6. [Convert nullptr to MLIR llvm-ptr<i8>](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L5786-L5793)
7. [Add FP80 ty, used in random number generator](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L5890-L5892)
8. [Create LLVM-funcs for runtime operations](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/CGCall.cc#L1639-L1657)

## Cgeist New Features
1. [Obtain length and element type of variable length array](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L363)
2. [Handler for CXX-NULL-Ptr-Literal-Expr](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L618)
3. [Common routine for null initialization](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L699). Called by some simple implicit value initialization list expression.
4. [Mimic clang's store to one unit logic](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L798). The main purpose is to store value as an integral unit, the original logic will call corresponding downstream handlers.
5. [Initialization routine for arrays](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L845). Handles initialization lists and default constructor.
6. [Common logic for initialization list expression](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L1059). Memory not allocated in this handler. Supported target types: static array, record, union.
7. [Add Compound literal expression](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L1560)
8. [Add pointer sub-assign op logic and fix integer type matching](https://github.com/jiange91/Disagg-mlir/blob/remote/tools/cgeist/Lib/clang-mlir.cc#L3317-L3351)
9. Add disagg target file, deprecated now.
10. Added builtin-expressions:
    * is_constant_evaluated
    * addressof
    * trap, unreachable
    * nextafterf
    * constant_p
    * clzs, clz, clzl, clzll
    * ceil, ceilf, ceill, ceilf|16|128


## TODO:
1. Should have an universal initialization logic.
> related to Bugfix-3
2. TypeIdExpr logic 