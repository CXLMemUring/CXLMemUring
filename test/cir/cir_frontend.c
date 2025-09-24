#include <stdint.h>

// Simple loop to exercise CIR->CIRA->LLVM flow
void saxpy(float *x, float *y, int n, float a) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

