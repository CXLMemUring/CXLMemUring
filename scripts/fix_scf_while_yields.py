#!/usr/bin/env python3
"""Fix scf.while blocks by adding missing scf.yield terminators.

This script processes MLIR files containing scf.while operations and adds
scf.yield terminators to 'do' blocks that are missing them.
"""

import sys


def fix_scf_while_yields(text):
    """Add scf.yield statements to scf.while do blocks that are missing them."""
    lines = text.splitlines()

    # Find all '} do {' patterns
    do_block_info = []
    for i, line in enumerate(lines):
        if '} do {' in line:
            # Find the closing brace of this do block
            # The '} do {' itself has the opening brace for the do block
            # We need to find the matching closing brace
            depth = 1  # We start with the { from '} do {'
            end_idx = -1

            for j in range(i + 1, len(lines)):
                depth += lines[j].count('{') - lines[j].count('}')
                if depth == 0:
                    end_idx = j
                    break

            if end_idx != -1:
                do_block_info.append([i, end_idx])  # Use list so we can modify

    # Process from innermost to outermost (reverse order)
    # This ensures when we insert lines, we can adjust indices properly
    for idx in range(len(do_block_info) - 1, -1, -1):
        start_idx, end_idx = do_block_info[idx]

        # Check if the line before the closing brace has scf.yield
        # Find last non-empty line before closing brace
        last_content_idx = end_idx - 1
        while last_content_idx > start_idx and not lines[last_content_idx].strip():
            last_content_idx -= 1

        if last_content_idx > start_idx:
            last_line = lines[last_content_idx].strip()
            if 'scf.yield' not in last_line:
                # Need to add scf.yield
                # Get indentation from the '} do {' line and add 2 spaces
                base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
                yield_indent = base_indent + 2
                yield_line = ' ' * yield_indent + 'scf.yield'

                # Insert before the closing brace
                lines.insert(end_idx, yield_line)

                # Adjust end_idx for all unprocessed blocks that end at or after the insertion point
                for i in range(idx):
                    if do_block_info[i][1] >= end_idx:
                        do_block_info[i][1] += 1

    return '\n'.join(lines)


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
