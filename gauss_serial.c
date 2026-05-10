#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024  // 矩阵大小，可根据需要调整

// 初始化矩阵（对称正定矩阵，保证可解）
void init_matrix(double *A, double *b, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i * n + j] = 100.0;  // 对角线放大，避免主元为0
            } else {
                A[i * n + j] = (rand() % 10) / 10.0;
            }
            sum += A[i * n + j];
        }
        b[i] = sum;  // 保证解向量x为全1
    }
}

// 串行高斯消去法（无选主元，仅基础实现）
void gauss_serial(double *A, double *b, double *x, int n) {
    // 前向消元
    for (int k = 0; k < n - 1; k++) {
        for (int i = k + 1; i < n; i++) {
            double factor = A[i * n + k] / A[k * n + k];
            for (int j = k; j < n; j++) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // 回代求解
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
    double *A = (double *)malloc(n * n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    init_matrix(A, b, n);

    clock_t start = clock();
    gauss_serial(A, b, x, n);
    clock_t end = clock();

    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("串行版执行时间: %.4f 秒\n", time);

    // 验证结果（打印前3个和后3个解）
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