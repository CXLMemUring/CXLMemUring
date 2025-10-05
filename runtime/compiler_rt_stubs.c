/* Compiler runtime stubs for aarch64 cross-compilation */

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