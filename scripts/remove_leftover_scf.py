#!/usr/bin/env python3
"""Remove leftover scf.yield and cf.br operations that should have been converted.

After SCF->CF->LLVM conversion, sometimes orphaned scf.yield and cf.br operations remain in the IR.
These are typically redundant when llvm.br already exists, and can be safely removed.
"""

import sys
import re

def is_llvm_terminator(line):
    """Check if a line is an LLVM terminator operation."""
    stripped = line.strip()
    # LLVM terminators: return, br, cond_br, unreachable, invoke, resume, switch, indirectbr
    return bool(re.match(r'^llvm\.(return|br|cond_br|unreachable|invoke|resume|switch|indirectbr)\b', stripped))

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


def remove_duplicate_terminators(lines):
    """Remove LLVM terminators that appear after another terminator in the same block.

    C++ exception handling patterns can produce code where llvm.unreachable follows
    llvm.return in the same basic block. This is invalid LLVM IR. We detect this by
    looking for consecutive terminator operations (not separated by a block label or
    closing brace) and removing the extra ones.
    """
    output = []
    prev_was_terminator = False
    for line in lines:
        stripped = line.strip()

        # Block boundaries reset the terminator flag
        if stripped.startswith('^') or stripped == '}' or stripped.startswith('llvm.func'):
            prev_was_terminator = False
            output.append(line)
            continue

        # Check if this line is a terminator
        if is_llvm_terminator(line):
            if prev_was_terminator:
                # Skip this line - it's a duplicate terminator
                continue
            prev_was_terminator = True
        else:
            # Non-terminator resets the flag (but only for non-empty lines)
            if stripped and not stripped.startswith('//') and not stripped.startswith('#'):
                prev_was_terminator = False

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
    # Also remove duplicate terminators (e.g., llvm.unreachable after llvm.return)
    cleaned = remove_duplicate_terminators(cleaned)

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
