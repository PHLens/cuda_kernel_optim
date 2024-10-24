#include <stdio.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels.cuh"
#include "utils.cuh"

#define MY_GEMM sgemm1_1

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

  const int BLOCK_SIZE_M = 32;
  const int BLOCK_SIZE_N = 32;
  const int BLOCK_SIZE_K = 32;

  float *d_A, *d_B, *d_C, *d_C1;
  checkCudaErrors(cudaMalloc(&d_A, size_A));
  checkCudaErrors(cudaMalloc(&d_B, size_B));
  checkCudaErrors(cudaMalloc(&d_C, size_C));
  checkCudaErrors(cudaMalloc(&d_C1, size_C));

  // generate data
  for (int i = 0; i < M * K; i++) {
    A[i] = i / 13;
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = i % 13;
  }

  checkCudaErrors(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_C1, C1, size_C, cudaMemcpyHostToDevice));

  cudaEvent_t s, e;
  checkCudaErrors(cudaEventCreate(&s));
  checkCudaErrors(cudaEventCreate(&e));
  float total_time_ms = 0.0;
  int nIters = 100;

  checkCudaErrors(cudaEventRecord(s));

  for (int i = 0; i < nIters; i++) {
    // Define the block size and grid size
    dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 gridDim((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    MY_GEMM<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
  }

  checkCudaErrors(cudaEventRecord(e));
  checkCudaErrors(cudaEventSynchronize(e));
  checkCudaErrors(cudaEventElapsedTime(&total_time_ms, s, e));
  checkCudaErrors(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

  printf("My kernel average time: %f ms.\n", total_time_ms / nIters);

  // cublas
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;

  checkCudaErrors(cudaEventRecord(s));

  for (int i = 0; i < nIters; i++) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C1, N);
  }

  checkCudaErrors(cudaEventRecord(e));
  checkCudaErrors(cudaEventSynchronize(e));
  checkCudaErrors(cudaEventElapsedTime(&total_time_ms, s, e));
  checkCudaErrors(cudaMemcpy(C1, d_C1, size_C, cudaMemcpyDeviceToHost));

  printf("CuBlas kernel average time: %f ms.\n", total_time_ms / nIters);

  cublasDestroy(handle);

  double eps = 1.e-6;
  isEqualT(C, C1, M, N, eps);

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaFree(d_C1));
  checkCudaErrors(cudaEventDestroy(s));
  checkCudaErrors(cudaEventDestroy(e));

  free(A);
  free(B);
  free(C);
  free(C1);
  return 0;
}
