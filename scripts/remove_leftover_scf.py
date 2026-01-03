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

def is_cir_type(type_str):
    """Check if a type string is a CIR type."""
    # CIR types: !cir.*, !u32i, !s32i, !u64i, !s64i, !s8i, !u8i, etc.
    return (type_str.startswith('!cir.') or
            re.match(r'^![us]\d+i$', type_str) or
            type_str == '!cir.bool')


def collect_cast_substitutions_per_function(lines):
    """Collect variable substitutions from unrealized_conversion_cast operations, per function.

    Returns a list of (start_line, end_line, substitutions) tuples for each function.
    """
    result = []
    current_func_start = 0
    current_substitutions = {}

    # Pattern: %result = builtin.unrealized_conversion_cast %input : type to type
    # Types can be either !mlir.dialect or basic types like i1, i32, i64
    cast_pattern = re.compile(r'^\s*(%\w+)\s*=\s*builtin\.unrealized_conversion_cast\s+(%\w+)\s*:\s*([^\s]+)\s+to\s+([^\s]+)')

    for i, line in enumerate(lines):
        # Detect function start
        if ('llvm.func' in line or 'cir.func' in line) and '{' in line:
            # Save previous function's substitutions
            if current_substitutions:
                result.append((current_func_start, i, current_substitutions))
            current_func_start = i
            current_substitutions = {}

        match = cast_pattern.match(line)
        if match:
            result_var = match.group(1)
            input_var = match.group(2)
            to_type = match.group(4)
            if is_cir_type(to_type):
                current_substitutions[result_var] = input_var

    # Add last function
    if current_substitutions:
        result.append((current_func_start, len(lines), current_substitutions))

    return result


def apply_substitutions_in_range(lines, func_ranges):
    """Apply substitutions only within their respective function ranges."""
    output = list(lines)  # Make a copy

    for start, end, substitutions in func_ranges:
        for i in range(start, min(end, len(output))):
            line = output[i]
            for old_var, new_var in substitutions.items():
                line = re.sub(r'\b' + re.escape(old_var) + r'\b', new_var, line)
            output[i] = line

    return output


def apply_substitutions(line, substitutions):
    """Replace variable references according to substitution map."""
    for old_var, new_var in substitutions.items():
        # Replace the variable when used as an operand
        # Use negative lookbehind for word chars before %, and word boundary after the digits
        # %157 should match but not %1570 or something%157
        line = re.sub(r'(?<![%\w])' + re.escape(old_var) + r'\b', new_var, line)
    return line


def convert_cir_type_to_llvm(type_str):
    """Convert a CIR type string to LLVM type string."""
    # Common conversions
    type_str = re.sub(r'!cir\.ptr<[^>]*>', '!llvm.ptr', type_str)
    type_str = re.sub(r'!cir\.bool', 'i8', type_str)  # CIR bool is typically i8 in LLVM
    type_str = re.sub(r'!s8i', 'i8', type_str)
    type_str = re.sub(r'!u8i', 'i8', type_str)
    type_str = re.sub(r'!s16i', 'i16', type_str)
    type_str = re.sub(r'!u16i', 'i16', type_str)
    type_str = re.sub(r'!s32i', 'i32', type_str)
    type_str = re.sub(r'!u32i', 'i32', type_str)
    type_str = re.sub(r'!s64i', 'i64', type_str)
    type_str = re.sub(r'!u64i', 'i64', type_str)
    type_str = re.sub(r'!cir\.array<[^>]*>', '!llvm.ptr', type_str)
    type_str = re.sub(r'!rec_\w+', '!llvm.ptr', type_str)
    return type_str


def cleanup_scf_ops(lines, counter_start=0):
    """Remove standalone scf.yield and convert cf.br to llvm.br."""
    # Counter for generating unique intermediate variable names
    counter = [counter_start]
    def unique_name(prefix):
        counter[0] += 1
        return f'%_{prefix}_{counter[0]}'

    # Pattern for unrealized_conversion_cast
    cast_pattern = re.compile(r'^\s*(%\w+)\s*=\s*builtin\.unrealized_conversion_cast\s+(%\w+)\s*:\s*([^\s]+)\s+to\s+([^\s]+)')

    # Track current function's substitutions (reset at function boundaries)
    current_substitutions = {}
    output = []
    for line in lines:
        # Reset substitutions at function boundaries
        if ('llvm.func' in line or 'cir.func' in line) and '{' in line:
            current_substitutions = {}

        # Collect substitutions from unrealized_conversion_cast
        cast_match = cast_pattern.match(line)
        if cast_match:
            result_var = cast_match.group(1)
            input_var = cast_match.group(2)
            to_type = cast_match.group(4)
            if is_cir_type(to_type):
                current_substitutions[result_var] = input_var
                # Don't output this line - it will be deleted
                continue

        # Apply all substitutions
        line = apply_substitutions(line, current_substitutions)

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
        # Convert cir.const to llvm.mlir.zero for null pointers
        cir_const_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.const\s+#cir\.ptr<null>\s*:\s*!cir\.ptr<[^>]*>$', line)
        if cir_const_match:
            indent = cir_const_match.group(1)
            var = cir_const_match.group(2)
            output.append(f'{indent}{var} = llvm.mlir.zero : !llvm.ptr')
            continue
        # Convert cir.br to llvm.br (CIR branch that wasn't converted to LLVM)
        cir_br_match = re.match(r'^(\s*)cir\.br\s+(.*)$', line)
        if cir_br_match:
            indent = cir_br_match.group(1)
            rest = cir_br_match.group(2)
            # Convert CIR type syntax to LLVM type syntax if present
            # cir.ptr<!void> -> !llvm.ptr, etc.
            rest = re.sub(r'!cir\.ptr<[^>]*>', '!llvm.ptr', rest)
            rest = re.sub(r'!u32i', 'i32', rest)
            rest = re.sub(r'!s32i', 'i32', rest)
            output.append(f'{indent}llvm.br {rest}')
            continue
        # Convert cir.brcond to llvm.cond_br
        cir_brcond_match = re.match(r'^(\s*)cir\.brcond\s+(%\w+)\s+(\^bb\d+),\s*(\^bb\d+)$', line)
        if cir_brcond_match:
            indent = cir_brcond_match.group(1)
            cond = cir_brcond_match.group(2)
            true_dest = cir_brcond_match.group(3)
            false_dest = cir_brcond_match.group(4)
            output.append(f'{indent}llvm.cond_br {cond}, {true_dest}, {false_dest}')
            continue
        # Convert cir.eh.inflight_exception to llvm.landingpad
        cir_eh_match = re.match(r'^(\s*)(%\w+),\s*(%\w+)\s*=\s*cir\.eh\.inflight_exception\s+cleanup\s*$', line)
        if cir_eh_match:
            indent = cir_eh_match.group(1)
            ptr_var = cir_eh_match.group(2)
            sel_var = cir_eh_match.group(3)
            lp = unique_name('landing_pad')
            output.append(f'{indent}{lp} = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>')
            output.append(f'{indent}{ptr_var} = llvm.extractvalue {lp}[0] : !llvm.struct<(ptr, i32)>')
            output.append(f'{indent}{sel_var} = llvm.extractvalue {lp}[1] : !llvm.struct<(ptr, i32)>')
            continue
        # Convert cir.resume to llvm.resume
        cir_resume_match = re.match(r'^(\s*)cir\.resume\s+(%\w+),\s*(%\w+)\s*$', line)
        if cir_resume_match:
            indent = cir_resume_match.group(1)
            ptr_var = cir_resume_match.group(2)
            sel_var = cir_resume_match.group(3)
            rs = unique_name('resume_struct')
            rs0 = unique_name('resume_struct')
            rs1 = unique_name('resume_struct')
            output.append(f'{indent}{rs} = llvm.mlir.poison : !llvm.struct<(ptr, i32)>')
            output.append(f'{indent}{rs0} = llvm.insertvalue {ptr_var}, {rs}[0] : !llvm.struct<(ptr, i32)>')
            output.append(f'{indent}{rs1} = llvm.insertvalue {sel_var}, {rs0}[1] : !llvm.struct<(ptr, i32)>')
            output.append(f'{indent}llvm.resume {rs1} : !llvm.struct<(ptr, i32)>')
            continue
        # Convert cf.cond_br to llvm.cond_br if present
        cf_cond_br_match = re.match(r'^(\s*)cf\.cond_br\s+(.*)$', line)
        if cf_cond_br_match:
            indent = cf_cond_br_match.group(1)
            rest = cf_cond_br_match.group(2)
            output.append(f'{indent}llvm.cond_br {rest}')
            continue

        # Convert cir.load to llvm.load (with optional align)
        # Pattern 1: with align - cir.load align(N) %ptr : type, type
        cir_load_align_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.load\s+align\((\d+)\)\s+(%\w+)\s*:\s*([^,]+),\s*(.+)$', line)
        if cir_load_align_match:
            indent = cir_load_align_match.group(1)
            result = cir_load_align_match.group(2)
            align = cir_load_align_match.group(3)
            ptr = cir_load_align_match.group(4)
            result_type = convert_cir_type_to_llvm(cir_load_align_match.group(6).strip())
            output.append(f'{indent}{result} = llvm.load {ptr} {{alignment = {align} : i64}} : !llvm.ptr -> {result_type}')
            continue

        # Pattern 2: without align - cir.load %ptr : type, type
        cir_load_noalign_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.load\s+(%\w+)\s*:\s*([^,]+),\s*(.+)$', line)
        if cir_load_noalign_match:
            indent = cir_load_noalign_match.group(1)
            result = cir_load_noalign_match.group(2)
            ptr = cir_load_noalign_match.group(3)
            result_type = convert_cir_type_to_llvm(cir_load_noalign_match.group(5).strip())
            # Use alignment 1 as default
            output.append(f'{indent}{result} = llvm.load {ptr} {{alignment = 1 : i64}} : !llvm.ptr -> {result_type}')
            continue

        # Convert cir.store to llvm.store
        cir_store_match = re.match(r'^(\s*)cir\.store\s+align\((\d+)\)\s+(%\w+),\s*(%\w+)\s*:\s*([^,]+),\s*(.+)$', line)
        if cir_store_match:
            indent = cir_store_match.group(1)
            align = cir_store_match.group(2)
            value = cir_store_match.group(3)
            ptr = cir_store_match.group(4)
            value_type = convert_cir_type_to_llvm(cir_store_match.group(5).strip())
            # ptr_type = cir_store_match.group(6)
            output.append(f'{indent}llvm.store {value}, {ptr} {{alignment = {align} : i64}} : {value_type}, !llvm.ptr')
            continue

        # Convert cir.unary(not, ...) to llvm.xor with 1 (boolean negation)
        cir_unary_not_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.unary\(not,\s*(%\w+)\)\s*:\s*!cir\.bool,\s*!cir\.bool$', line)
        if cir_unary_not_match:
            indent = cir_unary_not_match.group(1)
            result = cir_unary_not_match.group(2)
            operand = cir_unary_not_match.group(3)
            # XOR with 1 to negate a boolean, then produce i1 result for use in conditionals
            one = unique_name('one')
            xor_res = unique_name('xor_result')
            output.append(f'{indent}{one} = llvm.mlir.constant(1 : i8) : i8')
            output.append(f'{indent}{xor_res} = llvm.xor {operand}, {one}  : i8')
            output.append(f'{indent}{result} = llvm.trunc {xor_res} : i8 to i1')
            continue

        # Convert cir.call to llvm.call (with or without return value)
        # Pattern 1: void return - cir.call @func(...) : (...) -> () extra(...)
        cir_call_void_match = re.match(r'^(\s*)cir\.call\s+(@\w+)\(([^)]*)\)\s*:\s*\(([^)]*)\)\s*->\s*\(\)\s*extra\([^)]*\)$', line)
        if cir_call_void_match:
            indent = cir_call_void_match.group(1)
            func = cir_call_void_match.group(2)
            args = cir_call_void_match.group(3)
            arg_types = cir_call_void_match.group(4)
            llvm_arg_types = convert_cir_type_to_llvm(arg_types)
            output.append(f'{indent}llvm.call {func}({args}) : ({llvm_arg_types}) -> ()')
            continue

        # Pattern 2: non-void return - %result = cir.call @func(...) : (...) -> type extra(...)
        cir_call_ret_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.call\s+(@\w+)\(([^)]*)\)\s*:\s*\(([^)]*)\)\s*->\s*([^e][^\s]*)\s*extra\([^)]*\)$', line)
        if cir_call_ret_match:
            indent = cir_call_ret_match.group(1)
            result = cir_call_ret_match.group(2)
            func = cir_call_ret_match.group(3)
            args = cir_call_ret_match.group(4)
            arg_types = cir_call_ret_match.group(5)
            ret_type = cir_call_ret_match.group(6)
            llvm_arg_types = convert_cir_type_to_llvm(arg_types)
            llvm_ret_type = convert_cir_type_to_llvm(ret_type)
            output.append(f'{indent}{result} = llvm.call {func}({args}) : ({llvm_arg_types}) -> {llvm_ret_type}')
            continue

        # Convert cir.try_call to llvm.invoke (for exception handling)
        cir_try_call_match = re.match(r'^(\s*)(?:(%\w+)\s*=\s*)?cir\.try_call\s+(@\w+)\(([^)]*)\)\s+(\^bb\d+),\s*(\^bb\d+)\s*:\s*\(([^)]*)\)\s*->\s*(.+)$', line)
        if cir_try_call_match:
            indent = cir_try_call_match.group(1)
            result = cir_try_call_match.group(2)  # May be None
            func = cir_try_call_match.group(3)
            args = cir_try_call_match.group(4)
            normal_dest = cir_try_call_match.group(5)
            unwind_dest = cir_try_call_match.group(6)
            arg_types = convert_cir_type_to_llvm(cir_try_call_match.group(7))
            ret_type = convert_cir_type_to_llvm(cir_try_call_match.group(8).strip())
            if result and ret_type != '()':
                output.append(f'{indent}{result} = llvm.invoke {func}({args}) to {normal_dest} unwind {unwind_dest} : ({arg_types}) -> {ret_type}')
            else:
                output.append(f'{indent}llvm.invoke {func}({args}) to {normal_dest} unwind {unwind_dest} : ({arg_types}) -> ()')
            continue

        # Convert cir.return to llvm.return
        cir_return_match = re.match(r'^(\s*)cir\.return\s+(%\w+)\s*:\s*(.+)$', line)
        if cir_return_match:
            indent = cir_return_match.group(1)
            value = cir_return_match.group(2)
            orig_type = cir_return_match.group(3).strip()
            ret_type = convert_cir_type_to_llvm(orig_type)
            # If returning a bool (which is i8 in our conversion), truncate to i1
            # since function signatures typically use i1 for booleans
            if orig_type == '!cir.bool':
                trunc_var = unique_name('ret_bool')
                output.append(f'{indent}{trunc_var} = llvm.trunc {value} : i8 to i1')
                output.append(f'{indent}llvm.return {trunc_var} : i1')
            else:
                output.append(f'{indent}llvm.return {value} : {ret_type}')
            continue

        # Convert cir.return without value
        cir_return_void_match = re.match(r'^(\s*)cir\.return\s*$', line)
        if cir_return_void_match:
            indent = cir_return_void_match.group(1)
            output.append(f'{indent}llvm.return')
            continue

        # Convert cir.const #true / #false to llvm.mlir.constant
        cir_const_bool_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.const\s+#(true|false)\s*$', line)
        if cir_const_bool_match:
            indent = cir_const_bool_match.group(1)
            result = cir_const_bool_match.group(2)
            value = '1' if cir_const_bool_match.group(3) == 'true' else '0'
            output.append(f'{indent}{result} = llvm.mlir.constant({value} : i8) : i8')
            continue

        # Convert cir.get_global to llvm.mlir.addressof
        cir_get_global_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.get_global\s+(@[^\s:]+)\s*:\s*(.+)$', line)
        if cir_get_global_match:
            indent = cir_get_global_match.group(1)
            result = cir_get_global_match.group(2)
            global_name = cir_get_global_match.group(3)
            # global_type = cir_get_global_match.group(4)
            output.append(f'{indent}{result} = llvm.mlir.addressof {global_name} : !llvm.ptr')
            continue

        # Convert cir.cast(array_to_ptrdecay, ...) - this is just a pointer, no-op in LLVM
        cir_cast_decay_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.cast\(array_to_ptrdecay,\s*(%\w+)\s*:\s*[^)]+\),\s*.+$', line)
        if cir_cast_decay_match:
            indent = cir_cast_decay_match.group(1)
            result = cir_cast_decay_match.group(2)
            source = cir_cast_decay_match.group(3)
            # array_to_ptrdecay is essentially a no-op for pointers in LLVM
            # Just substitute result with source in later uses
            current_substitutions[result] = source
            continue

        # Convert cir.base_class_addr to llvm.getelementptr
        cir_base_class_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.base_class_addr\s+(%\w+)\s*:\s*[^\[]+\[(\d+)\]\s*->\s*.+$', line)
        if cir_base_class_match:
            indent = cir_base_class_match.group(1)
            result = cir_base_class_match.group(2)
            source = cir_base_class_match.group(3)
            offset = cir_base_class_match.group(4)
            output.append(f'{indent}{result} = llvm.getelementptr inbounds {source}[{offset}] : (!llvm.ptr) -> !llvm.ptr, i8')
            continue

        # Convert cir.select to llvm.select
        cir_select_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.select\s+if\s+(%\w+)\s+then\s+(%\w+)\s+else\s+(%\w+)\s*:\s*\([^)]+\)\s*->\s*(.+)$', line)
        if cir_select_match:
            indent = cir_select_match.group(1)
            result = cir_select_match.group(2)
            cond = cir_select_match.group(3)
            true_val = cir_select_match.group(4)
            false_val = cir_select_match.group(5)
            result_type = convert_cir_type_to_llvm(cir_select_match.group(6).strip())
            # Need to convert bool (i8) to i1 for llvm.select
            # llvm.select syntax: %res = llvm.select %cond, %true, %false : i1, result_type
            cond_i1 = unique_name('cond_i1')
            output.append(f'{indent}{cond_i1} = llvm.trunc {cond} : i8 to i1')
            output.append(f'{indent}{result} = llvm.select {cond_i1}, {true_val}, {false_val} : i1, {result_type}')
            continue

        # Convert cir.get_member to pointer arithmetic
        # Since we don't have struct layout info, we use byte offsets
        # For index 0, offset is 0 so result = source pointer
        # For other indices, we approximate using 8-byte alignment (pointer-sized on 64-bit)
        cir_get_member_match = re.match(r'^(\s*)(%\w+)\s*=\s*cir\.get_member\s+(%\w+)\[(\d+)\]\s*\{name\s*=\s*"[^"]*"\}\s*:\s*[^\-]+->\s*.+$', line)
        if cir_get_member_match:
            indent = cir_get_member_match.group(1)
            result = cir_get_member_match.group(2)
            source = cir_get_member_match.group(3)
            index = int(cir_get_member_match.group(4))
            if index == 0:
                # First member is at offset 0, just use the same pointer
                current_substitutions[result] = source
            else:
                # For other members, use byte offset (assume 8-byte alignment)
                byte_offset = index * 8
                output.append(f'{indent}{result} = llvm.getelementptr inbounds {source}[{byte_offset}] : (!llvm.ptr) -> !llvm.ptr, i8')
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

def fix_cond_br_types(lines):
    """Fix llvm.cond_br that uses i8 values instead of i1.

    When booleans are loaded as i8 but used in cond_br (which expects i1),
    we need to insert truncation operations. Track per-function to avoid
    name collisions across functions.
    """
    output = []
    counter = [0]
    # Track i8 values per function (reset at each function definition)
    i8_values = set()

    for i, line in enumerate(lines):
        # Reset tracking at function boundaries
        if 'llvm.func' in line and '{' in line:
            i8_values = set()

        # Match: %var = llvm.load ... : !llvm.ptr -> i8
        load_match = re.match(r'^\s*(%\w+)\s*=\s*llvm\.load\s+.*:\s*!llvm\.ptr\s*->\s*i8\s*$', line)
        if load_match:
            i8_values.add(load_match.group(1))

        # Match definitions that produce i1 (shouldn't be truncated)
        # These produce i1: icmp, fcmp, trunc to i1, etc.
        i1_def_match = re.match(r'^\s*(%\w+)\s*=\s*llvm\.(icmp|fcmp|trunc\s+.*:\s*\S+\s+to\s+i1)', line)
        if i1_def_match:
            # Remove from i8_values if present (same name was used for i1 result)
            i8_values.discard(i1_def_match.group(1))

        # Match: llvm.cond_br %cond, ^dest1, ^dest2
        cond_br_match = re.match(r'^(\s*)llvm\.cond_br\s+(%\w+),\s*(.*)$', line)
        if cond_br_match:
            indent = cond_br_match.group(1)
            cond = cond_br_match.group(2)
            rest = cond_br_match.group(3)
            if cond in i8_values:
                counter[0] += 1
                trunc_var = f'%_cond_trunc_{counter[0]}'
                output.append(f'{indent}{trunc_var} = llvm.trunc {cond} : i8 to i1')
                output.append(f'{indent}llvm.cond_br {trunc_var}, {rest}')
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
    # Run cleanup multiple times to handle newly collected substitutions
    prev_len = len(lines) + 1
    for _ in range(5):  # Max 5 iterations
        cleaned = cleanup_scf_ops(lines)
        cleaned = remove_duplicate_terminators(cleaned)
        if len(cleaned) >= prev_len:
            break  # No more progress
        prev_len = len(cleaned)
        lines = cleaned

    # Fix cond_br type mismatches
    cleaned = fix_cond_br_types(cleaned)

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
