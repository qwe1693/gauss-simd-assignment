#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>  // AVX2 头文件

#define N 1024

void init_matrix(double *A, double *b, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = 100.0;
            } else {
                A[i * n + j] = (rand() % 10) / 10.0;
            }
            sum += A[i * n + j];
        }
        b[i] = sum;
    }
}

// AVX2 加速的前向消元
void gauss_simd(double *A, double *b, double *x, int n) {
    // 前向消元（SIMD优化）
    for (int k = 0; k < n - 1; k++) {
        double akk = A[k * n + k];
        for (int i = k + 1; i < n; i++) {
            double factor = A[i * n + k] / akk;
            __m256d factor_vec = _mm256_set1_pd(factor);

            // 向量化循环（每次处理4个double）
            int j;
            for (j = k; j + 3 < n; j += 4) {
                __m256d Aik_vec = _mm256_loadu_pd(&A[i * n + j]);
                __m256d Akj_vec = _mm256_loadu_pd(&A[k * n + j]);
                __m256d res = _mm256_sub_pd(Aik_vec, _mm256_mul_pd(factor_vec, Akj_vec));
                _mm256_storeu_pd(&A[i * n + j], res);
            }
            // 处理剩余元素
            for (; j < n; j++) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // 回代求解（SIMD收益低，用串行即可）
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i * n + j] * x[j];
        }
        x[i] /= A[i * n + i];
    }
}

int main() {
    int n = N;
    double *A = (double *)aligned_alloc(32, n * n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    init_matrix(A, b, n);

    clock_t start = clock();
    gauss_simd(A, b, x, n);
    clock_t end = clock();

    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("SIMD版执行时间: %.4f 秒\n", time);

    printf("解向量验证:\n");
    for (int i = 0; i < 3; i++) {
        printf("x[%d] = %.6f\n", i, x[i]);
    }
    for (int i = n - 3; i < n; i++) {
        printf("x[%d] = %.6f\n", i, x[i]);
    }

    free(A);
    free(b);
    free(x);
    return 0;
}