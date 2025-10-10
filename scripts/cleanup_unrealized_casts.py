#!/usr/bin/env python3
"""Rewrite unrealized conversion casts (memref <-> llvm.ptr) emitted by lowering.

We look for the common pattern produced by the pipeline:

  %alloc = llvm.alloca ... -> !llvm.ptr
  ... build an LLVM memref descriptor (series of llvm.insertvalue)
  %memdesc = builtin.unrealized_conversion_cast %struct : !llvm.struct<...> to memref<...>
  %ptr = builtin.unrealized_conversion_cast %memdesc : memref<...> to !llvm.ptr

The descriptor is unused except for the bridge back to a pointer. Translation to
LLVM IR rejects the struct->memref cast. We replace any use of %ptr with the
original %alloc value and erase both unrealized casts. The descriptor itself is
left intact (it becomes dead and can be DCE'd later).
"""

from __future__ import annotations
import re
import sys

ALLOC_RE = re.compile(r"\s*(%[0-9A-Za-z_]+)\s*=\s*llvm\.alloca")
CAST_STRUCT_RE = re.compile(r"\s*(%[0-9A-Za-z_]+)\s*=\s*builtin\.unrealized_conversion_cast\s+(%[0-9A-Za-z_]+)\s*:\s*!llvm\.struct")
CAST_PTR_RE = re.compile(r"\s*(%[0-9A-Za-z_]+)\s*=\s*builtin\.unrealized_conversion_cast\s+(%[0-9A-Za-z_]+)\s*:\s*memref")
FUNC_RE = re.compile(r"\s*llvm\.func\s+@([A-Za-z0-9_]+)")
CAST_INDEX_RE = re.compile(r"\s*(%[0-9A-Za-z_]+)\s*=\s*builtin\.unrealized_conversion_cast\s+(%[0-9A-Za-z_]+)\s*:\s*i64\s+to\s+index")
# Match any cast that produces a memref type (struct -> memref)
CAST_TO_MEMREF_RE = re.compile(r"\s*(%[0-9A-Za-z_]+)\s*=\s*builtin\.unrealized_conversion_cast\s+.*to\s+memref")


def strip_module_attrs(lines: list[str]) -> list[str]:
    if not lines:
        return lines
    first = lines[0]
    attr_pos = first.find(' attributes')
    if attr_pos == -1:
        return lines
    brace_pos = first.find('{', attr_pos)
    if brace_pos == -1:
        return lines
    depth = 1
    j = brace_pos + 1
    while depth > 0 and j < len(first):
        if first[j] == '{':
            depth += 1
        elif first[j] == '}':
            depth -= 1
        j += 1
    new_first = first[:attr_pos].rstrip() + first[j:]
    lines[0] = new_first
    return lines


def cleanup(lines: list[str]) -> list[str]:
    lines = strip_module_attrs(lines)
    replacements: dict[tuple[str | None, str], str] = {}
    to_remove: set[int] = set()
    current_func: str | None = None

    # pass 1: collect replacements
    for idx, line in enumerate(lines):
        m_func = FUNC_RE.match(line)
        if m_func:
            current_func = m_func.group(1)

        m_ptr = CAST_PTR_RE.match(line)
        if m_ptr:
            ptr_var, memref_var = m_ptr.groups()

            # Find the struct->memref cast defining memref_var
            struct_idx = None
            for j in range(idx - 1, -1, -1):
                m_struct = CAST_STRUCT_RE.match(lines[j])
                if m_struct and m_struct.group(1) == memref_var:
                    struct_idx = j
                    break
                if lines[j].strip().startswith('llvm.func'):
                    break
            if struct_idx is None:
                continue

            # Find the closest preceding llvm.alloca in the same block/function
            alloca_var = None
            for j in range(struct_idx - 1, -1, -1):
                if lines[j].strip().startswith('}'):  # hit end of block
                    break
                m_alloc = ALLOC_RE.match(lines[j])
                if m_alloc:
                    alloca_var = m_alloc.group(1)
                    break
            if alloca_var is None:
                continue

            replacements[(current_func, ptr_var)] = alloca_var
            to_remove.add(idx)          # remove memref->ptr cast line
            to_remove.add(struct_idx)   # remove struct->memref cast line
            continue

        # Handle i64 -> index cast rewrite
        m_index = CAST_INDEX_RE.match(line)
        if m_index:
            idx_var, src_var = m_index.groups()
            replacements[(current_func, idx_var)] = src_var
            to_remove.add(idx)

        # Handle dead struct->memref casts (these can't be translated to LLVM IR)
        m_to_memref = CAST_TO_MEMREF_RE.match(line)
        if m_to_memref:
            memref_var = m_to_memref.group(1)
            # Check if this value is ever used (look ahead in the same function)
            used = False
            for j in range(idx + 1, len(lines)):
                if lines[j].strip().startswith('llvm.func') or lines[j].strip().startswith('func.func'):
                    break
                # Check if memref_var appears as an operand (not as a result)
                if re.search(r'(?<![0-9A-Za-z_])' + re.escape(memref_var) + r'(?![0-9A-Za-z_])', lines[j]):
                    # Make sure it's not the definition line
                    if not re.match(r"\s*" + re.escape(memref_var) + r"\s*=", lines[j]):
                        used = True
                        break
            if not used:
                to_remove.add(idx)

    # pass 2: emit lines applying replacements (function-scoped)
    output: list[str] = []
    current_func = None
    for idx, line in enumerate(lines):
        if idx in to_remove:
            continue
        m_func = FUNC_RE.match(line)
        if m_func:
            current_func = m_func.group(1)

        new_line = line
        for (func_name, old_var), new_var in replacements.items():
            if func_name == current_func and old_var in new_line:
                # Avoid rewriting the defining line of new_var itself
                if re.match(r"\s*" + re.escape(old_var) + r"\s*=", new_line):
                    continue
                new_line = re.sub(r"(?<![0-9A-Za-z_])" + re.escape(old_var) + r"(?![0-9A-Za-z_])",
                                   new_var, new_line)
        output.append(new_line)
    return output


def main():
    if len(sys.argv) > 1:
        text = open(sys.argv[1]).read()
    else:
        text = sys.stdin.read()
    lines = text.splitlines()
    cleaned = cleanup(lines)
    sys.stdout.write("\n".join(cleaned) + ("\n" if cleaned and not cleaned[-1].endswith("\n") else ""))


if __name__ == "__main__":
    main()
