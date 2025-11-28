#!/usr/bin/env python3
"""
Cleanup script to remove CIR dialect remnants from LLVM MLIR output.

This handles:
1. CIR type aliases (lines starting with ! that reference !cir.)
2. Dead blocks (blocks with "no predecessors" that contain CIR operations)
3. Unrealized conversion casts to CIR types
4. CIR operations that weren't converted
5. CIR attributes on the module
"""

import re
import sys

def cleanup_cir_remnants(input_path, output_path=None):
    with open(input_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    output_lines = []

    # Track if we're inside a dead block with CIR operations
    in_dead_block = False
    dead_block_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip CIR type alias definitions at the top of the file
        # These are lines like: !rec_xxx = !cir.record<...>
        if re.match(r'^![a-zA-Z0-9_]+ = !cir\.', line):
            i += 1
            continue

        # Skip CIR attribute definitions
        if re.match(r'^#[a-zA-Z0-9_]+ = #cir', line):
            i += 1
            continue

        # Check for dead block markers
        if '// no predecessors' in line:
            # This is a dead block - skip it and all following lines until the next block
            in_dead_block = True
            # Get the indentation level
            match = re.match(r'^(\s*)\^bb\d+:', line)
            if match:
                dead_block_indent = len(match.group(1))
            i += 1
            continue

        # If we're in a dead block, skip until we hit the next block or end of function
        if in_dead_block:
            # Check if this is a new block or end of function
            if re.match(r'^\s*\^bb\d+:', line) or re.match(r'^\s*\}', line) or re.match(r'^\s*llvm\.func', line):
                in_dead_block = False
                # Don't skip this line - process it normally
            else:
                i += 1
                continue

        # Skip lines with cir. operations
        if re.search(r'\bcir\.(call|store|load|yield|br|scope)', line):
            i += 1
            continue

        # Skip unrealized_conversion_cast to CIR types
        if 'unrealized_conversion_cast' in line and '!cir.' in line:
            i += 1
            continue

        # Clean up module attributes - remove CIR-specific attributes
        if 'module @' in line:
            # Remove cir.lang, cir.sob, cir.opt_info, cir.type_size_info, cir.uwtable attributes
            line = re.sub(r',?\s*cir\.lang\s*=\s*#cir\.lang<[^>]*>', '', line)
            line = re.sub(r',?\s*cir\.sob\s*=\s*#cir\.signed_overflow_behavior<[^>]*>', '', line)
            line = re.sub(r',?\s*cir\.opt_info\s*=\s*#cir\.opt_info<[^>]*>', '', line)
            line = re.sub(r',?\s*cir\.type_size_info\s*=\s*#cir\.type_size_info<[^>]*>', '', line)
            line = re.sub(r',?\s*cir\.uwtable\s*=\s*#cir\.uwtable<[^>]*>', '', line)
            # Remove cir.triple (should be llvm.target_triple after conversion)
            line = re.sub(r',?\s*cir\.triple\s*=\s*"[^"]*"', '', line)
            # Clean up any double commas or trailing commas before }
            line = re.sub(r',\s*,', ',', line)
            line = re.sub(r',\s*\}', '}', line)
            line = re.sub(r'\{\s*,', '{', line)

        # Skip lines that reference CIR types in operations
        if re.search(r':\s*!cir\.', line) or re.search(r'->\s*!cir\.', line):
            i += 1
            continue

        # Skip extra(#fn_attr) on operations
        line = re.sub(r'\s*extra\(#[a-zA-Z0-9_]+\)', '', line)

        output_lines.append(line)
        i += 1

    result = '\n'.join(output_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(result)
    else:
        print(result)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: cleanup_cir_remnants.py <input.mlir> [output.mlir]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    cleanup_cir_remnants(input_path, output_path)
