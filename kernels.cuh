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
