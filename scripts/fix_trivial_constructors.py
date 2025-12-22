#!/usr/bin/env python3
"""
Fix C++ functions in LLVM IR that are either:
1. Declared but not defined (trivial constructors that should be empty)
2. Defined but broken (libstdc++ internals that need real implementations)
3. LLVM intrinsics with naming issues

This handles:
1. Trivial constructors (STL tag types, empty structs) -> add empty definitions
2. Broken definitions (ios_base, locale, etc.) -> convert to declarations
3. LLVM intrinsics naming fixes (va_start.p0 -> va_start)
"""

import re
import sys

# Patterns for functions that should NOT be defined - they must be external
# These are libstdc++ implementation details that need their real implementations
BROKEN_DEFINITIONS_PATTERNS = [
    # ios_base constructor - needs proper initialization
    r'_ZNSt8ios_baseC\dEv',
    # basic_ios constructors
    r'_ZNSt9basic_iosIcSt11char_traitsIcEEC\dEv',
    # basic_ostream constructors
    r'_ZNSoC\dE',
    r'_ZNSt13basic_ostreamIcSt11char_traitsIcEEC\dE',
    # basic_istream constructors
    r'_ZNSiC\dE',
    r'_ZNSt13basic_istreamIcSt11char_traitsIcEEC\dE',
    # locale constructors and operations
    r'_ZNSt6localeC\dEv',
    r'_ZNSt6localeC\dERKS_',
    r'_ZNSt6localeD\dEv',
    r'_ZNSt6localeaSERKS_',
    # Note: std::__cxx11::basic_string methods that need replacement (like default
    # constructors and _M_init_local_buf) are handled by NEEDS_REPLACEMENT_PATTERNS
    # below, not removed here
]

# Patterns for trivial functions that should be empty (void return, ptr arg)
TRIVIAL_VOID_PTR_PATTERNS = [
    # STL iterator tags
    r'_ZNSt\d*random_access_iterator_tagC\dEv',
    r'_ZNSt\d*forward_iterator_tagC\dEv',
    r'_ZNSt\d*bidirectional_iterator_tagC\dEv',
    r'_ZNSt\d*input_iterator_tagC\dEv',
    r'_ZNSt\d*output_iterator_tagC\dEv',
    # STL internal types
    r'_ZNSt\d*_Enable_default_constructor_tagC\dEv',
    r'_ZNSt\d*_Index_tuple.*C\dEv',
    r'_ZNSt\d*piecewise_construct_tC\dEv',
    r'_ZNSt\d*tuple.*C\dEv',  # empty tuple constructors
    # GNU C++ ops (match various namespace depths)
    r'_ZN9__gnu_cxx\d*__ops\d*_Val_less_iterC\dEv',
    r'_ZN9__gnu_cxx\d*__ops\d*_Iter_less_iterC\dEv',
    r'_ZN9__gnu_cxx\d*__ops\d*_Iter_less_valC\dEv',
    r'_ZN9__gnu_cxx\d*__ops\d*_Iter_equal_to_iterC\dEv',
    r'_ZN9__gnu_cxx5__ops14_Val_less_iterC\dEv',
    r'_ZN9__gnu_cxx5__ops19_Iter_equal_to_iterC\dEv',
    # std::greater and other functors (match various template instantiations)
    r'_ZNSt7greaterI.*EC\dEv',
    r'_ZNSt4lessI.*EC\dEv',
    r'_ZNSt8equal_toI.*EC\dEv',
    # Note: _M_init_local_buf is NOT trivial - it needs to set the data pointer
    # to the local buffer. The generated code is broken and needs fixing differently.
    # allocator_traits default construct
    r'_ZNSt16allocator_traitsISaI.*EE9constructI.*EEvRS\d_PT_DpOT0_',
]

# Pattern to match void(ptr) declarations
DECLARE_VOID_PTR_PATTERN = re.compile(r'^declare\s+void\s+@([^\s(]+)\s*\(ptr[^)]*\)\s*.*$')

# Pattern to match function definitions (any return type) and capture return type
DEFINE_PATTERN = re.compile(r'^define\s+(?:\w+\s+)*(void|ptr|i\d+|float|double|%[^\s]+)\s+@([^\s(]+)\s*\(([^)]*)\)')

# Patterns for functions with broken implementations that need replacement
# These are generated with empty or wrong bodies by ClangIR
NEEDS_REPLACEMENT_PATTERNS = [
    r'_ZNSt7__cxx1112basic_stringI.*E17_M_init_local_bufEv',
    # Replace default string constructors entirely
    r'_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC[12]Ev',
    # Copy constructor C1 is broken - does raw memcpy which breaks SSO
    r'_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1ERKS4_',
]

def needs_replacement(name):
    """Check if function needs its implementation replaced."""
    for pattern in NEEDS_REPLACEMENT_PATTERNS:
        if re.match(pattern, name):
            return True
    return False

def get_replacement_impl(name):
    """Get replacement implementation for a broken function."""
    if '_M_init_local_bufEv' in name:
        # _M_init_local_buf needs to set _M_dataplus._M_p to point to local buffer
        # In std::__cxx11::basic_string layout:
        #   offset 0: _M_dataplus._M_p (ptr to char data)
        #   offset 8: _M_string_length
        #   offset 16: _M_local_buf[16] (the SSO buffer)
        # This function should do: this->_M_dataplus._M_p = &this->_M_local_buf
        return f'''define linkonce_odr void @{name}(ptr %0) {{
  ; Set _M_dataplus._M_p (offset 0) to point to _M_local_buf (offset 16)
  %local_buf = getelementptr inbounds i8, ptr %0, i64 16
  store ptr %local_buf, ptr %0, align 8
  ret void
}}'''
    # Replace default constructor with a fully correct implementation
    if re.match(r'_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC[12]Ev', name):
        return f'''define linkonce_odr void @{name}(ptr %0) {{
  ; std::string default constructor
  ; Layout: [0]: _M_p, [8]: length, [16]: local_buf[16]

  ; Set _M_p to point to local buffer (offset 16)
  %local_buf = getelementptr inbounds i8, ptr %0, i64 16
  store ptr %local_buf, ptr %0, align 8

  ; Set length to 0 (offset 8)
  %len_ptr = getelementptr inbounds i8, ptr %0, i64 8
  store i64 0, ptr %len_ptr, align 8

  ; Set null terminator at local_buf[0]
  store i8 0, ptr %local_buf, align 1

  ret void
}}'''
    # Replace copy constructor - the generated one does raw memcpy which breaks SSO
    # For SSO strings, _M_p points to offset 16 within the object itself
    # After memcpy, dest._M_p still points to source+16, not dest+16
    if re.match(r'_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1ERKS4_', name):
        return f'''define linkonce_odr void @{name}(ptr %0, ptr %1) {{
  ; std::string copy constructor - must handle SSO correctly
  ; Layout: [0]: _M_p (ptr), [8]: length (i64), [16]: local_buf[16]
  ;
  ; If source uses SSO (_M_p == source+16), we need to:
  ;   1. Set dest._M_p = dest+16
  ;   2. Copy the data to dest's local buffer
  ; If source uses heap, we need to call the real allocating copy

  ; Get source's _M_p
  %src_p = load ptr, ptr %1, align 8

  ; Calculate source's local_buf address (source + 16)
  %src_local_buf = getelementptr inbounds i8, ptr %1, i64 16

  ; Check if source is using SSO: _M_p == &_M_local_buf
  %is_sso = icmp eq ptr %src_p, %src_local_buf
  br i1 %is_sso, label %sso_path, label %heap_path

sso_path:
  ; SSO: copy 32 bytes then fix up _M_p
  call void @llvm.memcpy.p0.p0.i64(ptr %0, ptr %1, i64 32, i1 false)

  ; Fix _M_p to point to dest's local buffer
  %dest_local_buf = getelementptr inbounds i8, ptr %0, i64 16
  store ptr %dest_local_buf, ptr %0, align 8
  br label %done

heap_path:
  ; Heap allocation: call the C2 constructor which handles allocation properly
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2ERKS4_(ptr %0, ptr %1)
  br label %done

done:
  ret void
}}'''
    return None

def is_trivial_function(name):
    """Check if the function name matches a trivial function pattern."""
    for pattern in TRIVIAL_VOID_PTR_PATTERNS:
        if re.match(pattern, name):
            return True
    return False

def is_broken_definition(name):
    """Check if the function is a broken definition that should be external."""
    for pattern in BROKEN_DEFINITIONS_PATTERNS:
        if re.match(pattern, name):
            return True
    return False

def fix_intrinsics(line):
    """Fix LLVM intrinsic naming issues."""
    # Fix llvm.va_start.p0 -> llvm.va_start
    line = re.sub(r'@llvm\.va_start\.p0\b', '@llvm.va_start', line)
    line = re.sub(r'@llvm\.va_end\.p0\b', '@llvm.va_end', line)
    line = re.sub(r'@llvm\.va_copy\.p0\.p0\b', '@llvm.va_copy', line)
    return line

def fix_linkage(line):
    """Fix available_externally linkage to linkonce_odr for inline functions."""
    # available_externally means "use external definition" which may not exist
    # Change to linkonce_odr to provide the definition
    if 'define available_externally' in line:
        line = line.replace('define available_externally', 'define linkonce_odr')
    return line

# Pattern to match stores of std::string aggregates
STRING_STORE_PATTERN = re.compile(
    r'^\s*store\s+%"class\.std::__cxx11::basic_string<char>"\s+%(\w+),\s+ptr\s+%(\w+),\s+align\s+8\s*$'
)

# Pattern to match memcpy of std::string (32 bytes = string struct size)
STRING_MEMCPY_PATTERN = re.compile(
    r'^\s*call\s+void\s+@llvm\.memcpy\.p0\.p0\.\w+\s*\(\s*ptr\s+%(\w+),\s+ptr\s+%(\w+),\s+i32\s+32,\s+i1\s+false\s*\)\s*$'
)

# Counter for generating unique SSA names
_fixup_counter = [0]

def get_sso_fixup_code(dest_ptr_name):
    """Generate code to fix string pointer after aggregate store.

    After a raw aggregate store:
    - SSO strings: _M_p points to source's local buffer (wrong)
    - Heap strings: _M_p points to shared heap memory (causes double-free)

    We call a helper function that properly copies the string data.
    """
    n = _fixup_counter[0]
    _fixup_counter[0] += 1
    return f'''  ; String copy fixup for %{dest_ptr_name}
  call void @__fix_string_after_aggregate_store(ptr %{dest_ptr_name})'''


# Helper function definition to add at the end of the file
STRING_FIXUP_HELPER = '''
; Helper function to fix std::string after aggregate store
; This ensures proper copying of both SSO and heap-allocated strings
define internal void @__fix_string_after_aggregate_store(ptr %str) {
entry:
  ; Layout: [0]: _M_p (ptr), [8]: length (i64), [16]: local_buf[16]
  %len_ptr = getelementptr inbounds i8, ptr %str, i64 8
  %len = load i64, ptr %len_ptr, align 8
  %is_sso = icmp ule i64 %len, 15
  br i1 %is_sso, label %sso, label %heap

sso:
  ; For SSO: just fix _M_p to point to our local buffer
  %local_buf = getelementptr inbounds i8, ptr %str, i64 16
  store ptr %local_buf, ptr %str, align 8
  ret void

heap:
  ; For heap: allocate new memory and copy
  ; new_size = len + 1 (for null terminator)
  %new_size = add i64 %len, 1
  %new_ptr = call ptr @malloc(i64 %new_size)
  %old_ptr = load ptr, ptr %str, align 8
  ; Copy len+1 bytes (including null terminator)
  call void @llvm.memcpy.p0.p0.i64(ptr %new_ptr, ptr %old_ptr, i64 %new_size, i1 false)
  ; Update _M_p to point to new memory
  store ptr %new_ptr, ptr %str, align 8
  ret void
}

; Declare external functions used by helper
declare ptr @malloc(i64)
'''

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.ll>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]

    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()
    lines = content.split('\n')

    # Reset counter for each file
    _fixup_counter[0] = 0

    # Track which declarations we've converted to definitions
    trivial_converted = set()
    definitions_to_add = []

    # Track broken definitions to remove
    broken_defs_to_remove = set()

    # Track definitions that need replacement
    replacement_defs = {}

    # First pass: identify declarations to convert to trivial definitions
    for line in lines:
        match = DECLARE_VOID_PTR_PATTERN.match(line.strip())
        if match:
            func_name = match.group(1)
            if is_trivial_function(func_name):
                trivial_converted.add(func_name)
                # Create empty function definition
                definitions_to_add.append(
                    f'define linkonce_odr void @{func_name}(ptr %0) {{\n'
                    f'  ret void\n'
                    f'}}'
                )

    # Also identify broken definitions to remove, and definitions needing replacement
    # Store full signature info for broken definitions
    broken_def_signatures = {}
    for line in lines:
        match = DEFINE_PATTERN.match(line.strip())
        if match:
            ret_type = match.group(1)
            func_name = match.group(2)
            params = match.group(3)
            if is_broken_definition(func_name):
                broken_defs_to_remove.add(func_name)
                # Strip parameter names, keep only types
                param_types = []
                for p in params.split(','):
                    p = p.strip()
                    if p:
                        # Extract type (first word or ptr/i32/etc)
                        parts = p.split()
                        if parts:
                            param_types.append(parts[0])
                broken_def_signatures[func_name] = (ret_type, ', '.join(param_types))
            elif needs_replacement(func_name):
                replacement = get_replacement_impl(func_name)
                if replacement:
                    replacement_defs[func_name] = replacement

    # Second pass: output with fixes
    i = 0
    while i < len(lines):
        line = lines[i]

        # Fix intrinsic naming
        line = fix_intrinsics(line)
        # Fix available_externally linkage
        line = fix_linkage(line)

        # Check if this is a declaration we're converting to definition
        match = DECLARE_VOID_PTR_PATTERN.match(line.strip())
        if match and match.group(1) in trivial_converted:
            # Skip the declaration, we'll add the definition later
            i += 1
            continue

        # Check if this is a broken definition to remove or replace
        match = DEFINE_PATTERN.match(line.strip())
        if match:
            func_name = match.group(2)  # group(1) is return type, group(2) is name
            if func_name in broken_defs_to_remove:
                # Skip the entire function definition (until closing brace)
                brace_count = 0
                while i < len(lines):
                    if '{' in lines[i]:
                        brace_count += lines[i].count('{')
                    if '}' in lines[i]:
                        brace_count -= lines[i].count('}')
                    i += 1
                    if brace_count == 0:
                        break
                # Add a declaration instead with proper signature
                ret_type, param_types = broken_def_signatures.get(func_name, ('void', 'ptr'))
                print(f'declare {ret_type} @{func_name}({param_types})')
                continue
            elif func_name in replacement_defs:
                # Skip the entire function definition (until closing brace)
                brace_count = 0
                while i < len(lines):
                    if '{' in lines[i]:
                        brace_count += lines[i].count('{')
                    if '}' in lines[i]:
                        brace_count -= lines[i].count('}')
                    i += 1
                    if brace_count == 0:
                        break
                # Add replacement implementation
                print(replacement_defs[func_name])
                continue

        # Check for std::string aggregate stores that need SSO fixup
        store_match = STRING_STORE_PATTERN.match(line)
        if store_match:
            src_name = store_match.group(1)
            dest_ptr_name = store_match.group(2)
            print(line)
            print(get_sso_fixup_code(dest_ptr_name))
            i += 1
            continue

        # Check for memcpy of std::string structs (32 bytes)
        memcpy_match = STRING_MEMCPY_PATTERN.match(line)
        if memcpy_match:
            dest_ptr_name = memcpy_match.group(1)
            # src_ptr_name = memcpy_match.group(2)
            print(line)
            print(get_sso_fixup_code(dest_ptr_name))
            i += 1
            continue

        print(line)
        i += 1

    # Add the generated trivial definitions at the end
    for defn in definitions_to_add:
        print(defn)

    # Add the string fixup helper if we inserted any fixup calls
    if _fixup_counter[0] > 0:
        print(STRING_FIXUP_HELPER)

if __name__ == '__main__':
    main()
