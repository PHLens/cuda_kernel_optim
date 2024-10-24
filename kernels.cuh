// naive kernel
__global__ void sgemm0(float* A, float* B, float* C, const int M, const int N, const int K) {
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

// global memory tiled to shared memory
template < const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K >
__global__ void sgemm1(float* A, float* B, float* C, const int M, const int N, const int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * BLOCK_SIZE_M + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE_N + threadIdx.x;
  __shared__ float tileA[BLOCK_SIZE_M][BLOCK_SIZE_N];
  __shared__ float tileB[BLOCK_SIZE_M][BLOCK_SIZE_N];

  float c = 0.0;
  for (int m = 0; m < K / BLOCK_SIZE_K; m++) {
    // parallel loading M/N into shared memory.
    tileA[ty][tx] = A[row * K + m * BLOCK_SIZE_K + tx];
    tileB[ty][tx] = B[(m * BLOCK_SIZE_K + ty) * K + col];
  
    __syncthreads();
    for (int i = 0; i < BLOCK_SIZE_K; i++) {
      c += tileA[ty][i] * tileB[i][tx];
    }
    __syncthreads();
  }

  C[row * N + col] = c;
}

// handle boundary conditions. padding with 0.
template < const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K >
__global__ void sgemm1_1(float* A, float* B, float* C, const int M, const int N, const int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * BLOCK_SIZE_M + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE_N + threadIdx.x;
  __shared__ float tileA[BLOCK_SIZE_M][BLOCK_SIZE_N];
  __shared__ float tileB[BLOCK_SIZE_M][BLOCK_SIZE_N];

  float c = 0.0;
  for (int m = 0; m < (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; m++) {
    // parallel loading M/N into shared memory.
    if (row < M && m * BLOCK_SIZE_K + tx < M) {
      tileA[ty][tx] = A[row * K + m * BLOCK_SIZE_K + tx];
    } else {
      tileA[ty][tx] = 0;
    }

    if (m * BLOCK_SIZE_K + ty < N && col < N) {
      tileB[ty][tx] = B[(m * BLOCK_SIZE_K + ty) * K + col];
    } else {
      tileB[ty][tx] = 0;
    }
  
    __syncthreads();
    if (row < M && col < N) {
      for (int i = 0; i < BLOCK_SIZE_K; i++) {
        c += tileA[ty][i] * tileB[i][tx];
      }
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = c;
  }
}
