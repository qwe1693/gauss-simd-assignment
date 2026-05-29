#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#define N 1024

// 和你原代码完全一致的初始化
void init_matrix(double *A, double *b, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) A[i * n + j] = 100.0;
            else A[i * n + j] = (rand() % 10) / 10.0;
            sum += A[i * n + j];
        }
        b[i] = sum;
    }
}

// OpenMP+SIMD 融合并行
void gauss_omp_simd(double *A, double *b, double *x, int n) {
    for (int k = 0; k < n - 1; k++) {
        double akk = A[k * n + k];
        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; i++) {
            double factor = A[i * n + k] / akk;
            __m256d factor_vec = _mm256_set1_pd(factor);
            int j;
            // SIMD 内层循环（每次处理4个double）
            for (j = k; j + 3 < n; j += 4) {
                __m256d Aik = _mm256_loadu_pd(&A[i * n + j]);
                __m256d Akj = _mm256_loadu_pd(&A[k * n + j]);
                __m256d res = _mm256_sub_pd(Aik, _mm256_mul_pd(factor_vec, Akj));
                _mm256_storeu_pd(&A[i * n + j], res);
            }
            // 剩余元素串行处理
            for (; j < n; j++) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
    // 回代（串行）
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i * n + j] * x[j];
        }
        x[i] /= A[i * n + i];
    }
}

int main() {
    double *A = (double*)aligned_alloc(32, N * N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *x = (double*)malloc(N * sizeof(double));
    init_matrix(A, b, N);

    double start = omp_get_wtime();
    gauss_omp_simd(A, b, x, N);
    double end = omp_get_wtime();
    int nth = omp_get_max_threads();
    printf("OpenMP+SIMD版（%d线程）执行时间: %.4f 秒\n", nth, end - start);

    // 验证解向量
    printf("解向量验证:\n");
    for (int i = 0; i < 3; i++) {
        printf("x[%d] = %.6f\n", i, x[i]);
    }
    for (int i = N - 3; i < N; i++) {
        printf("x[%d] = %.6f\n", i, x[i]);
    }

    free(A); free(b); free(x);
    return 0;
}