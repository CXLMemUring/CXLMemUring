//===- cgeist.cpp - Minimal cgeist Driver ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for cgeist when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  std::cout << "cgeist - placeholder version" << std::endl;
  std::cout << "This is a minimal version for build testing only." << std::endl;
  
  std::cout << "Command line arguments:" << std::endl;
  for (int i = 0; i < argc; i++) {
    std::cout << "  " << i << ": " << argv[i] << std::endl;
  }
  
  return 0;
}
