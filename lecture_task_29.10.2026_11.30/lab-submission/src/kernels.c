/* Explicit loop orders, C-contiguous storage, IEEE double arithmetic. */
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

API double stride_sum(const double *data, size_t count, size_t stride) {
    double total = 0.0;
    for (size_t i = 0; i < count; i += stride) total += data[i];
    return total;
}

API void matmul_ijk(const double *a, const double *b, double *c, int n, int m) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
            double value = 0.0;
            for (int k = 0; k < m; ++k)
                value += a[(size_t)i*n+k] * b[(size_t)k*n+j];
            c[(size_t)i*m+j] = value;
        }
}

API void matmul_ikj(const double *a, const double *b, double *c, int n, int m) {
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < m; ++k) {
            const double aik = a[(size_t)i*n+k];
            for (int j = 0; j < m; ++j)
                c[(size_t)i*m+j] += aik * b[(size_t)k*n+j];
        }
}

/* Five-point average: four additions and one multiplication = 5 FLOPs/cell.
   Fixed zero boundaries; disjoint output rows; input never modified. */
API void stencil(const double *input, double *output, int n, int first, int last) {
    for (int i = first; i < last; ++i)
        for (int j = 1; j < n-1; ++j) {
            const size_t k = (size_t)i*n+j;
            output[k] = (input[k] + input[k-1] + input[k+1]
                         + input[k-n] + input[k+n]) * 0.2;
        }
}

/* Volatile prevents the compiler replacing K memory updates with one add. */
API void counter_loop(volatile int64_t *counters, int index, int64_t iterations) {
    for (int64_t i = 0; i < iterations; ++i) counters[index] += 1;
}
