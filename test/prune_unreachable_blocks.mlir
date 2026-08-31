// RUN: cira -allow-unregistered-dialect %s --cira-prune-unreachable-blocks | FileCheck %s

module {
  func.func @keeps_only_reachable(%condition: i1) {
    cf.cond_br %condition, ^left, ^exit
  ^left:
    cf.br ^exit
  ^dead:
    "test.unconverted"() : () -> ()
    cf.br ^exit
  ^exit:
    return
  }
}

// CHECK-LABEL: func.func @keeps_only_reachable
// CHECK: cf.cond_br
// CHECK: cf.br
// CHECK-NOT: test.unconverted
// CHECK: return
