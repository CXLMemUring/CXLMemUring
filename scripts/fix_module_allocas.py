#!/usr/bin/env python3
"""Move module-level allocas into the functions that use them.

Some CIR inputs contain allocas at module scope (outside any function) which is
invalid MLIR/LLVM IR. These allocas are used within functions, so we need to:

1. Identify all module-level alloca definitions (SSA values outside functions)
2. For each function, find which module-level SSA values it references
3. Create local copies of those allocas at the function entry
4. Replace all references within the function to use the local copies
5. Remove the now-unused module-level allocas

This script operates on MLIR text representation.
"""

from __future__ import annotations
import re
import sys
from typing import Dict, List, Set, Tuple


# Match module-level alloca definitions: %N = llvm.alloca or llvm.mlir.constant
# Note: \s* instead of \s+ to handle llvm.mlir.constant(1 : i64) where parenthesis follows immediately
MODULE_ALLOCA_RE = re.compile(r'^\s*(%\d+)\s*=\s*(llvm\.(?:alloca|mlir\.constant))\s*(.*)$')

# Match function definition opening
FUNC_START_RE = re.compile(r'^(\s*)(func\.func|llvm\.func)\s+(?:private\s+)?(?:internal\s+)?(?:@\S+).*\{')

# Match function ending (closing brace at appropriate indentation)
FUNC_END_RE = re.compile(r'^(\s*)\}')

# Match SSA value references (operands, not definitions)
SSA_REF_RE = re.compile(r'(?<![%0-9A-Za-z_])(%\d+)(?![0-9A-Za-z_])')


def find_module_allocas(lines: List[str]) -> Dict[str, Tuple[int, str, str]]:
    """Find all module-level alloca definitions.

    Returns dict mapping SSA name -> (line_idx, operation, operands)
    """
    allocas = {}
    in_function = False
    brace_depth = 0

    for idx, line in enumerate(lines):
        # Track if we're inside a function
        if FUNC_START_RE.match(line):
            in_function = True
            brace_depth = line.count('{') - line.count('}')
            continue

        if in_function:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0:
                in_function = False
            continue

        # We're at module level - check for allocas
        m = MODULE_ALLOCA_RE.match(line)
        if m:
            ssa_name, op, operands = m.groups()
            allocas[ssa_name] = (idx, op, operands.rstrip())

    return allocas


def find_function_regions(lines: List[str]) -> List[Tuple[int, int, str]]:
    """Find all function definitions.

    Returns list of (start_line, end_line, func_name)
    """
    regions = []
    func_start = None
    func_name = None
    brace_depth = 0

    for idx, line in enumerate(lines):
        if func_start is None:
            m = FUNC_START_RE.match(line)
            if m:
                # Extract function name
                name_match = re.search(r'@(\S+)', line)
                func_name = name_match.group(1) if name_match else "unknown"
                func_start = idx
                brace_depth = line.count('{') - line.count('}')
                if brace_depth <= 0:
                    # Single-line function or declaration
                    func_start = None
                continue
        else:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0:
                regions.append((func_start, idx, func_name))
                func_start = None

    return regions


def find_ssa_refs_in_line(line: str, exclude_def: bool = True) -> Set[str]:
    """Find all SSA value references in a line.

    If exclude_def=True, excludes the LHS definition (e.g., %x = ... won't include %x)
    """
    refs = set()

    # Check if this line has a definition on the LHS
    def_match = re.match(r'^\s*(%\d+)\s*=', line)
    defined_ssa = def_match.group(1) if def_match else None

    # Find all SSA references
    for m in SSA_REF_RE.finditer(line):
        ssa = m.group(1)
        if exclude_def and ssa == defined_ssa:
            continue
        refs.add(ssa)

    return refs


def process_file(lines: List[str]) -> List[str]:
    """Process the file to fix module-level allocas."""
    # Find module-level allocas
    module_allocas = find_module_allocas(lines)
    if not module_allocas:
        return lines

    # Find function regions
    functions = find_function_regions(lines)
    if not functions:
        return lines

    # For each function, find which module-level allocas it uses
    func_uses: Dict[Tuple[int, int], Set[str]] = {}
    for start, end, name in functions:
        uses = set()
        for idx in range(start, end + 1):
            refs = find_ssa_refs_in_line(lines[idx])
            for ref in refs:
                if ref in module_allocas:
                    uses.add(ref)
        if uses:
            func_uses[(start, end)] = uses

    # If no functions use module-level allocas, just remove them
    used_allocas = set()
    for uses in func_uses.values():
        used_allocas.update(uses)

    # Build output with modifications
    output = []
    lines_to_skip = set()

    # Mark module-level alloca lines for removal
    for ssa_name, (line_idx, op, operands) in module_allocas.items():
        lines_to_skip.add(line_idx)
        # Also skip the associated constant if it's on the previous line
        if line_idx > 0 and ssa_name in used_allocas:
            # Check if the previous line defines a constant used by this alloca
            prev_line = lines[line_idx - 1]
            if 'llvm.mlir.constant' in prev_line:
                # Extract the SSA defined on that line
                m = re.match(r'^\s*(%\d+)\s*=\s*llvm\.mlir\.constant', prev_line)
                if m:
                    const_ssa = m.group(1)
                    # Check if this constant is used in the alloca line
                    if const_ssa in lines[line_idx]:
                        lines_to_skip.add(line_idx - 1)

    # Process line by line
    for idx, line in enumerate(lines):
        if idx in lines_to_skip:
            continue

        # Check if this is the start of a function that uses module allocas
        found_func = None
        for (start, end), uses in func_uses.items():
            if idx == start:
                found_func = (start, end, uses)
                break

        if found_func:
            start, end, uses = found_func
            # Output the function opening line
            output.append(line)

            # Generate replacement allocas at function entry
            # Find proper indentation from next line
            next_idx = idx + 1
            while next_idx <= end and not lines[next_idx].strip():
                next_idx += 1
            base_indent = "    "  # Default
            if next_idx <= end:
                m = re.match(r'^(\s*)', lines[next_idx])
                if m:
                    base_indent = m.group(1)

            # Create a mapping from old SSA names to new SSA names
            # We need to find the next available SSA number in the function
            max_ssa = 0
            for fidx in range(start, end + 1):
                for m in re.finditer(r'%(\d+)', lines[fidx]):
                    max_ssa = max(max_ssa, int(m.group(1)))

            ssa_remap = {}
            next_ssa = max_ssa + 1

            # For each used alloca, create local versions
            for ssa_name in sorted(uses, key=lambda x: int(x[1:])):  # Sort by number
                line_idx, op, operands = module_allocas[ssa_name]

                if op == 'llvm.alloca':
                    # Check if the operand is a constant from the previous line
                    orig_line = lines[line_idx]
                    # Parse the alloca operands to find the count value
                    # Pattern: %N = llvm.alloca %M x type ...
                    alloca_match = re.match(r'.*llvm\.alloca\s+(%\d+)\s+x\s+(.+)', orig_line)
                    if alloca_match:
                        count_ssa = alloca_match.group(1)
                        rest = alloca_match.group(2)

                        # We need to create the constant first
                        new_const_ssa = f'%{next_ssa}'
                        next_ssa += 1
                        output.append(f'{base_indent}{new_const_ssa} = llvm.mlir.constant(1 : i64) : i64')

                        new_alloca_ssa = f'%{next_ssa}'
                        next_ssa += 1
                        output.append(f'{base_indent}{new_alloca_ssa} = llvm.alloca {new_const_ssa} x {rest}')
                        ssa_remap[ssa_name] = new_alloca_ssa
                    else:
                        # Simpler pattern
                        new_alloca_ssa = f'%{next_ssa}'
                        next_ssa += 1
                        output.append(f'{base_indent}{new_alloca_ssa} = {op} {operands}')
                        ssa_remap[ssa_name] = new_alloca_ssa
                elif op == 'llvm.mlir.constant':
                    new_ssa = f'%{next_ssa}'
                    next_ssa += 1
                    output.append(f'{base_indent}{new_ssa} = {op} {operands}')
                    ssa_remap[ssa_name] = new_ssa

            # Continue with the rest of the function, replacing SSA references
            for fidx in range(idx + 1, end + 1):
                func_line = lines[fidx]
                if fidx in lines_to_skip:
                    continue
                # Replace SSA references
                for old_ssa, new_ssa in ssa_remap.items():
                    # Don't replace the definition itself
                    if not re.match(rf'^\s*{re.escape(old_ssa)}\s*=', func_line):
                        func_line = re.sub(
                            rf'(?<![%0-9A-Za-z_]){re.escape(old_ssa)}(?![0-9A-Za-z_])',
                            new_ssa,
                            func_line
                        )
                output.append(func_line)

            # Skip to after the function
            idx = end
        else:
            # Check if we're inside a function region that we've already processed
            skip = False
            for (start, end), uses in func_uses.items():
                if start < idx <= end:
                    skip = True
                    break
            if not skip:
                output.append(line)

    return output


def main():
    if len(sys.argv) < 2:
        print("Usage: fix_module_allocas.py input.mlir [output.mlir]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(input_path) as f:
        lines = f.read().splitlines()

    result = process_file(lines)

    output_text = '\n'.join(result) + '\n'

    if output_path:
        with open(output_path, 'w') as f:
            f.write(output_text)
    else:
        sys.stdout.write(output_text)


if __name__ == '__main__':
    main()
