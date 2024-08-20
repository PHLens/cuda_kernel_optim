#include <stdio.h>
#include <stdlib.h>

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

// bool checkError(float* A, float* B) {
//   for (int i = 0; i < M * N; i++) {
//     double abs_err = fabs(A[i], B[i]);
//   }
// }
