#!/usr/bin/env python3
"""
Cleanup script to remove CIR dialect remnants from LLVM MLIR output.

This handles:
1. CIR type aliases (lines starting with ! that reference !cir.)
2. Dead blocks (blocks with "no predecessors" that contain CIR operations)
3. Unrealized conversion casts to CIR types
4. CIR operations that weren't converted
5. CIR attributes on the module
6. std::regex-related functions that can't be lowered properly (pre-processing)
"""

import re
import sys

# Patterns for std::regex-related function names that cause issues
# These match the mangled C++ names in the function symbol
REGEX_FUNC_PATTERNS = [
    # std::__cxx11::basic_regex methods (mangled: _ZNSt7__cxx1111basic_regex...)
    r'_ZNSt7__cxx1111basic_regex',
    r'_ZNKSt7__cxx1111basic_regex',
    # std::__cxx11::match_results methods (mangled: _ZNSt7__cxx1113match_results...)
    r'_ZNSt7__cxx1113match_results',
    r'_ZNKSt7__cxx1113match_results',
    # std::__cxx11::sub_match methods (mangled: _ZNSt7__cxx119sub_match...)
    r'_ZNSt7__cxx119sub_matchI',
    r'_ZNKSt7__cxx119sub_matchI',
    # std::__detail::_Executor (mangled: _ZNSt8__detail9_Executor...)
    r'_ZNSt8__detail9_Executor',
    r'_ZNKSt8__detail9_Executor',
    # std::__detail::_NFA (mangled: _ZNSt8__detail4_NFA...)
    r'_ZNSt8__detail4_NFA',
    r'_ZNKSt8__detail4_NFA',
    # std::__detail::_Scanner (mangled: _ZNSt8__detail8_Scanner...)
    r'_ZNSt8__detail8_Scanner',
    r'_ZNKSt8__detail8_Scanner',
    # std::__detail::_State (mangled: _ZNSt8__detail6_State...)
    r'_ZNSt8__detail6_State',
    # regex_match, regex_search, regex_replace (mangled: _ZSt11regex_match, etc.)
    r'_ZSt11regex_match',
    r'_ZSt12regex_search',
    r'_ZSt13regex_replace',
    # std::regex_traits (mangled: _ZNSt7__cxx1112regex_traits...)
    r'_ZNSt7__cxx1112regex_traits',
    r'_ZNKSt7__cxx1112regex_traits',
    # __shared_ptr for NFA types
    r'_ZN.*__shared_ptr.*_NFA',
    r'_ZNK.*__shared_ptr.*_NFA',
    # Container methods that operate on sub_match vectors (these are hard to convert)
    r'_ZNSt6vectorINSt7__cxx119sub_match',
    r'_ZNKSt6vectorINSt7__cxx119sub_match',
    r'_ZNSt12_Vector_baseINSt7__cxx119sub_match',
]

def is_regex_related_func(func_name):
    """Check if a function name is related to std::regex."""
    for pattern in REGEX_FUNC_PATTERNS:
        if re.search(pattern, func_name):
            return True
    return False

def preprocess_cir_for_regex(lines):
    """
    Pre-process CIR to remove/stub std::regex-related functions.
    Returns modified lines list.

    We only remove function bodies, not type aliases (those may be used by other code).
    """
    output_lines = []
    i = 0
    in_regex_func = False
    brace_depth = 0

    while i < len(lines):
        line = lines[i]

        # Always keep type alias definitions (lines starting with !)
        if re.match(r'^![a-zA-Z0-9_]+ = ', line):
            output_lines.append(line)
            i += 1
            continue

        # Always keep attribute definitions (lines starting with #)
        if re.match(r'^#[a-zA-Z0-9_]+ = ', line):
            output_lines.append(line)
            i += 1
            continue

        # Always keep location definitions
        if re.match(r'^#loc\d+ = ', line):
            output_lines.append(line)
            i += 1
            continue

        # Check for cir.func definition
        func_match = re.match(r'(\s*)cir\.func\s+.*@([A-Za-z0-9_]+)', line)
        if func_match and not in_regex_func:
            indent = func_match.group(1)
            func_name = func_match.group(2)

            if is_regex_related_func(func_name):
                # For regex functions, convert to external declaration (remove body)
                # This keeps the function declaration so calls are valid
                in_regex_func = True
                brace_depth = line.count('{') - line.count('}')

                # Extract the function signature and make it a private declaration
                # Remove the body - find where the signature ends and add just the declaration
                if '{' in line:
                    # Function has inline body starting on same line
                    # Convert: cir.func ... @name(...) -> type { ... }
                    # To:      cir.func private @name(...) -> type
                    sig_part = line.split('{')[0].strip()
                    # Remove any linkage specifiers and ensure just 'cir.func private'
                    sig_part = re.sub(r'\bcir\.func\s+(linkonce_odr\s+)?(comdat\s+)?(dso_local\s+)?(private\s+)?', 'cir.func private ', sig_part)
                    output_lines.append(sig_part)
                else:
                    # Declaration only - keep as is but ensure it's private
                    decl_line = re.sub(r'\bcir\.func\s+(linkonce_odr\s+)?(comdat\s+)?(dso_local\s+)?(private\s+)?', 'cir.func private ', line)
                    output_lines.append(decl_line)
                i += 1
                continue

        # If we're skipping a regex function
        if in_regex_func:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0:
                in_regex_func = False
            i += 1
            continue

        # Check for globals with regex-related names - skip them to avoid conversion issues
        # Note: cir.global in this file format is a single line ending with loc(#locN)
        if re.match(r'\s*cir\.global', line):
            # Extract the symbol name from the global
            global_match = re.search(r'@([A-Za-z0-9_]+)', line)
            if global_match:
                global_name = global_match.group(1)
                # Skip globals that are specifically regex-related by mangled name
                if is_regex_related_func(global_name):
                    i += 1
                    continue

        output_lines.append(line)
        i += 1

    return output_lines

def cleanup_cir_remnants(input_path, output_path=None, preprocess_regex=False):
    with open(input_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # If preprocessing for regex, ONLY run the regex preprocessor (no cleanup)
    # This produces a valid CIR file that can be passed to cir-opt
    if preprocess_regex:
        output_lines = preprocess_cir_for_regex(lines)
        result = '\n'.join(output_lines)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(result)
        else:
            print(result)
        return

    # Normal cleanup mode - for post-conversion LLVM MLIR files
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

def has_regex_usage(input_path):
    """Check if a CIR file has std::regex usage that would prevent lowering."""
    with open(input_path, 'r') as f:
        content = f.read()

    # Check for regex-related type definitions or function names
    regex_indicators = [
        r'std::__cxx11::basic_regex',
        r'std::__detail::_Executor',
        r'std::__detail::_NFA',
        r'std::__detail::_Scanner',
        r'std::__cxx11::match_results',
        r'std::__cxx11::sub_match',
        r'!rec_std3A3A__cxx113A3Abasic_regex',
        r'_ZNSt7__cxx1111basic_regex',
        r'_ZNSt8__detail9_Executor',
        r'regex_traits',
    ]

    for pattern in regex_indicators:
        if re.search(pattern, content):
            return True
    return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cleanup CIR remnants from MLIR files')
    parser.add_argument('input', help='Input MLIR file')
    parser.add_argument('output', nargs='?', help='Output MLIR file (stdout if not specified)')
    parser.add_argument('--preprocess-regex', action='store_true',
                       help='Pre-process to remove std::regex-related functions')
    parser.add_argument('--check-regex', action='store_true',
                       help='Check if file has std::regex usage (exit 1 if yes, 0 if no)')
    args = parser.parse_args()

    if args.check_regex:
        if has_regex_usage(args.input):
            print(f"File has std::regex usage: {args.input}", file=sys.stderr)
            sys.exit(1)
        else:
            sys.exit(0)

    cleanup_cir_remnants(args.input, args.output, args.preprocess_regex)
