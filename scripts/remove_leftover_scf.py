#!/usr/bin/env python3
"""Remove leftover scf.yield and cf.br operations that should have been converted.

After SCF->CF->LLVM conversion, sometimes orphaned scf.yield and cf.br operations remain in the IR.
These are typically redundant when llvm.br already exists, and can be safely removed.
"""

import sys
import re

def cleanup_scf_ops(lines):
    """Remove standalone scf.yield and convert cf.br to llvm.br."""
    output = []
    for line in lines:
        # Skip lines that are just scf.yield with optional leading whitespace
        if re.match(r'^\s*scf\.yield\s*$', line):
            continue
        # Convert cf.br to llvm.br (control flow branch that wasn't converted)
        cf_br_match = re.match(r'^(\s*)cf\.br\s+(.*)$', line)
        if cf_br_match:
            indent = cf_br_match.group(1)
            rest = cf_br_match.group(2)
            output.append(f'{indent}llvm.br {rest}')
            continue
        # Convert cf.cond_br to llvm.cond_br if present
        cf_cond_br_match = re.match(r'^(\s*)cf\.cond_br\s+(.*)$', line)
        if cf_cond_br_match:
            indent = cf_cond_br_match.group(1)
            rest = cf_cond_br_match.group(2)
            output.append(f'{indent}llvm.cond_br {rest}')
            continue
        output.append(line)
    return output

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    lines = text.splitlines()
    cleaned = cleanup_scf_ops(lines)

    # Write output
    output_text = '\n'.join(cleaned)
    if text.endswith('\n') and cleaned:
        output_text += '\n'

    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w') as f:
            f.write(output_text)
    else:
        sys.stdout.write(output_text)

if __name__ == '__main__':
    main()
