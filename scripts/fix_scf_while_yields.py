#!/usr/bin/env python3
"""Fix scf.while blocks by adding missing scf.yield terminators.

This script processes MLIR files containing scf.while operations and adds
scf.yield terminators to 'do' blocks that are missing them.
"""

import sys


def fix_scf_while_yields(text):
    """Add scf.yield statements to scf.while do blocks that are missing them."""
    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        result.append(line)

        # Look for } do {
        if '} do {' in line:
            # Track brace depth to find the matching closing }
            base_indent = len(line) - len(line.lstrip())
            depth = 1
            j = i + 1

            while j < len(lines) and depth > 0:
                if '{' in lines[j]:
                    depth += lines[j].count('{')
                if '}' in lines[j]:
                    depth -= lines[j].count('}')

                if depth == 0:
                    # Found the closing } of the do block
                    # Check if previous line already has scf.yield
                    prev_idx = len(result) - 1
                    while prev_idx >= 0 and not result[prev_idx].strip():
                        prev_idx -= 1

                    if prev_idx >= 0 and 'scf.yield' not in result[prev_idx]:
                        # Add scf.yield with proper indentation
                        yield_indent = base_indent + 2
                        result.append(' ' * yield_indent + 'scf.yield')

                    result.append(lines[j])
                    i = j
                    break
                else:
                    result.append(lines[j])
                j += 1

            if depth != 0:
                # Didn't find closing brace, malformed input
                # Just continue without modification
                i += 1
            else:
                i += 1
        else:
            i += 1

    return '\n'.join(result)


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    fixed = fix_scf_while_yields(text)

    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w') as f:
            f.write(fixed)
    else:
        sys.stdout.write(fixed)


if __name__ == "__main__":
    main()
