#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void sgemm(float* A, float* B, float* C, const int M, const int N, const int K) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < N && ty < M) {
    float c = 0.0;
    for (int i = 0; i < K; i++) {
      c += A[ty * K + i] * B[i * N + tx];
    }
    C[ty * N + tx] = c;
  }
}

void printM(float* C, int M, int N) {
  printf("[ ");
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (j == 0 && i != 0) printf("  ");
      if (j != N - 1) {
        printf("%f, ", C[i * N + j]);
      } else {
        printf("%f ", C[i * N + j]);
      }
    }
    if (i == M - 1) {
      printf("]\n");
    } else {
      printf("\n");
    }
  }
}


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

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  // generate data
  for (int i = 0; i < M * K; i++) {
    A[i] = i + 1;
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = i + 1;
  }
  printM(A, M, K);
  printM(B, K, N);

  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice);

  // Define the block size and grid size
  dim3 blockDim(16, 16);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  sgemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

  cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
  
  printM(C, M, N);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);
  return 0;
}
