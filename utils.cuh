#include <stdio.h>
#include <stdlib.h>

#define checkCudaErrors(func)                                                     \
{                                                                                 \
    cudaError_t error = (func);                                                       \
    if(error != cudaSuccess)                                                          \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(error));  \
}

void printM(float* C, int M, int N);

void isEqual(float* A, float* B, int M, int N, float eps);
void isEqualT(float* A, float* B_T, int M, int N, float eps);
