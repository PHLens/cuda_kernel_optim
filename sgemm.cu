#include <stdio.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels.cuh"
#include "utils.cuh"

#define MY_GEMM sgemm0

int main(int argc, const char* argv[]) {
  if (argc != 4) {
    printf("Arguments error! Usage: ./sgemm_gpu [M] [N] [K]\n");
    exit(0);
  }

  size_t M = atoi(argv[1]);
  size_t N = atoi(argv[2]);
  size_t K = atoi(argv[3]);

  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);
  float* A = (float*)malloc(size_A);
  float* B = (float*)malloc(size_B);
  float* C = (float*)malloc(size_C);
  float* C1 = (float*)malloc(size_C);

  float *d_A, *d_B, *d_C, *d_C1;
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);
  cudaMalloc(&d_C1, size_C);

  // generate data
  for (int i = 0; i < M * K; i++) {
    A[i] = i + 1;
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = i + 1;
  }
  // printf("A: ");
  // printM(A, M, K);
  // printf("B: ");
  // printM(B, K, N);

  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C1, C1, size_C, cudaMemcpyHostToDevice);

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  float total_time_ms = 0.0;
  int nIters = 10;

  cudaEventRecord(s);

  for (int i = 0; i < nIters; i++) {
    // Define the block size and grid size
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    MY_GEMM<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
  }

  cudaEventRecord(e);
  cudaEventSynchronize(e);
  cudaEventElapsedTime(&total_time_ms, s, e);
  cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

  // printf("C: ");
  // printM(C, M, N);
  printf("My kernel average time: %f ms.\n", total_time_ms / nIters);

  // cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;
  cudaEventRecord(s);
  for (int i = 0; i < nIters; i++) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C1, N);
  }

  cudaEventRecord(e);
  cudaEventSynchronize(e);
  cudaEventElapsedTime(&total_time_ms, s, e);
  cudaMemcpy(C1, d_C1, size_C, cudaMemcpyDeviceToHost);

  // printf("C1: ");
  // printM(C1, M, N);
  printf("CuBlas kernel average time: %f ms.\n", total_time_ms / nIters);

  cublasDestroy(handle); 

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C1);
  cudaEventDestroy(s);
  cudaEventDestroy(e);

  free(A);
  free(B);
  free(C);
  free(C1);
  return 0;
}
