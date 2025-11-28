/* Compiler runtime stubs for riscv64 cross-compilation */

/* Convert signed long to double */
double __floatdidf(long long a) {
    return (double)a;
}

/* Convert unsigned long to double */
double __floatundidf(unsigned long long a) {
    return (double)a;
}

/* Additional stubs if needed */
double __divdf3(double a, double b) {
    return a / b;
}

double __muldf3(double a, double b) {
    return a * b;
}

double __adddf3(double a, double b) {
    return a + b;
}

double __subdf3(double a, double b) {
    return a - b;
}

/* Integer multiplication/division for soft-float */
long long __muldi3(long long a, long long b) {
    return a * b;
}

long long __divdi3(long long a, long long b) {
    return a / b;
}

unsigned long long __udivdi3(unsigned long long a, unsigned long long b) {
    return a / b;
}

long long __moddi3(long long a, long long b) {
    return a % b;
}

unsigned long long __umoddi3(unsigned long long a, unsigned long long b) {
    return a % b;
}
