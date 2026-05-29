#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#define N 1024

// 全局变量（线程共享）
double *A, *b, *x;
int nth; // 线程数

// 初始化（和原代码完全一致）
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

// 线程函数：并行前向消元（行划分）
void* thread_func(void* arg) {
    int tid = *(int*)arg;
    free(arg);

    for (int k = 0; k < N - 1; k++) {
        // 行划分：i = k+1+tid, k+1+tid+nth, ...
        for (int i = k + 1 + tid; i < N; i += nth) {
            double factor = A[i * N + k] / A[k * N + k];
            for (int j = k; j < N; j++)
                A[i * N + j] -= factor * A[k * N + j];
            b[i] -= factor * b[k];
        }
    }
    return NULL;
}

// 串行回代（和原代码一致）
void back_substitute(double *A, double *b, double *x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++)
            x[i] -= A[i * n + j] * x[j];
        x[i] /= A[i * n + i];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) { printf("用法: %s <线程数>\n", argv[0]); return 1; }
    nth = atoi(argv[1]);

    // 分配内存（和原代码一致）
    A = (double*)malloc(N * N * sizeof(double));
    b = (double*)malloc(N * sizeof(double));
    x = (double*)malloc(N * sizeof(double));
    init_matrix(A, b, N);

    clock_t start = clock();

    // 并行前向消元
    pthread_t threads[nth];
    for (int t = 0; t < nth; t++) {
        int* tid = (int*)malloc(sizeof(int));
        *tid = t;
        pthread_create(&threads[t], NULL, thread_func, tid);
    }
    for (int t = 0; t < nth; t++) pthread_join(threads[t], NULL);

    // 串行回代
    back_substitute(A, b, x, N);

    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("pthread版（%d线程）执行时间: %.4f 秒\n", nth, time);

    // 验证（和原代码一致）
    printf("解向量验证:\n");
    for (int i = 0; i < 3; i++) printf("x[%d] = %.6f\n", i, x[i]);
    for (int i = N - 3; i < N; i++) printf("x[%d] = %.6f\n", i, x[i]);

    free(A); free(b); free(x);
    return 0;
}