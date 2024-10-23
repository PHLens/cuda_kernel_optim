#include "utils.cuh"


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

void isEqual(float* A, float* B, int M, int N, float eps) {
  bool eq = true;
  for (int i = 0; i < M * N; i++) {
    double abs_err = fabs(A[i] - B[i]);
    double rel_err = abs_err / fabs(A[i]) / M; // rel_err divided by M for large Matrix.
    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
              i, A[i], B[i], eps);
      eq = false;
      break;
    }
  }
  printf("%s\n", eq ? "Result SUCCESS!" : "Result FAILED!");
  // assert(eq);
}

void isEqualT(float* A, float* B_T, int M, int N, float eps) {
  bool eq = true;
  for (int i = 0; i < M * N; i++) {
    int row = i / N;
    int col = i % N;
    double abs_err = fabs(A[i] - B_T[col * M + row]);
    double rel_err = abs_err / fabs(A[i]) / M; // rel_err divided by M for large Matrix.
    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
              i, A[i], B_T[col * M + row], eps);
      eq = false;
      break;
    }
  }
  printf("%s\n", eq ? "Result SUCCESS!" : "Result FAILED!");
  // assert(eq);
}
