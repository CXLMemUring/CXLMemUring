module {
  func.func @translate_node_ptr(%node: !llvm.ptr) -> index {
    %paddr = cira.offload.get_paddr("next", %node) : !llvm.ptr -> index
    return %paddr : index
  }
}
