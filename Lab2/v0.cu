/*
* ECE1782 - W2024 - Lab 2 - Sample Code
* Sample Test Cases (sum)
n, result 
100,18295201.010496
200,147100808.124588
300,497296827.464880
400,1179763265.153962
500,2305380127.308517
600,3985027420.060339
700,6329585154.758305
800,9449933335.045414
*/

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code == cudaSuccess)
        return;

    fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
        exit(code);
}

/*Use the following to get a timestamp*/
double getTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

void iniData(float *B, int n)
{
    int i, j, k;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n; k++)
            {
                B[i * n * n + j * n + k] = (float)((i + j + k) % 10) * (float)1.1;
            }
        }
    }
    // for (i = 0; i < n; i++)
    // {
    //     for (j = 0; j < n; j++)
    //     {

    //         for (k = 0; k < n; k++)
    //         {
    //             printf("B: %f", B[i * n * n + j * n + k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}

void computeOnCPU(float *A, float *B, int n)
{
    int i, j, k;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n; k++)
            {
                float a = (i - 1 >= 0) ? B[(i - 1) * n * n + j * n + k] : 0;
                float b = (i + 1 < n) ? B[(i + 1) * n * n + j * n + k] : 0;
                float c = (j - 1 >= 0) ? B[i * n * n + (j - 1) * n + k] : 0;
                float d = (j + 1 < n) ? B[i * n * n + (j + 1) * n + k] : 0;
                float e = (k - 1 >= 0) ? B[i * n * n + j * n + k - 1] : 0;
                float f = (k + 1 < n) ? B[i * n * n + j * n + k + 1] : 0;
                A[i * n * n + j * n + k] = (float)0.8 * (a + b + c + d + e + f);
            }
        }
    }
}

__global__ void kernel(float *A, float *B, int n)
{

    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int z = blockIdx.z * blockDim.z + threadIdx.z;
    // int idx = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int idx = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
    // printf("idx %d\n", idx);
    int i = idx / (n * n);
    int z = idx % (n * n);
    int j = z / n;
    int k = z % n;
    if (i < n && j < n && k < n)
    {
        // printf("K: %d j: %d i: %d idx: %d\n", k, j, i, idx);
        float a = (i - 1 >= 0) ? B[(i - 1) * n * n + j * n + k] : 0;
        float b = (i + 1 < n) ? B[(i + 1) * n * n + j * n + k] : 0;
        float c = (j - 1 >= 0) ? B[i * n * n + (j - 1) * n + k] : 0;
        float d = (j + 1 < n) ? B[i * n * n + (j + 1) * n + k] : 0;
        float e = (k - 1 >= 0) ? B[i * n * n + j * n + k - 1] : 0;
        float f = (k + 1 < n) ? B[i * n * n + j * n + k + 1] : 0;
        A[i * n * n + j * n + k] = (float)0.8 * (a + b + c + d + e + f);
    }
}

bool checkEqual(float *h_A, float *h_dA, int n)
{
    float epsilon = 1e-6;
    int i, j, k;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n; k++)
            {
                // printf("da: %f ", h_dA[i * n * n + j * n + k]);
                if (fabsf(h_A[i * n * n + j * n + k] - h_dA[i * n * n + j * n + k]) > epsilon)
                {
                    printf("%d %d %d", i, j, k);
                    return false;
                }
            }
            // printf("\n");
        }
        // printf("\n");
    }
    return true;
}

double getSum(float *h_dA, int n)
{
    int i, j, k;
    double sum = 0;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < n; k++)
            {
                sum += h_dA[i * n * n + j * n + k] * (((i + j + k) % 10) ? 1 : -1);
            }
        }
    }
    return sum;
}

int main(int argc, char *argv[])
{

    // set matrix size
    if (argc != 2)
    {
        printf("Error: wrong number of args\n");
        exit(1);
    }

    int n = atoi(argv[1]);
    size_t number_of_elements = ((size_t)n) * n * n;
    size_t bytes = number_of_elements * sizeof(float);
    gpuErrchk(cudaDeviceReset());

    float *h_A, *h_B, *h_dA;
    gpuErrchk(cudaHostAlloc((void **)&h_A, bytes, 0));
    gpuErrchk(cudaHostAlloc((void **)&h_B, bytes, 0));
    gpuErrchk(cudaHostAlloc((void **)&h_dA, bytes, 0));
    iniData(h_B, n);

    double start_time = getTimeStamp();
    float *d_A, *d_B;
    gpuErrchk(cudaMalloc((void **)&d_A, bytes));
    gpuErrchk(cudaMalloc((void **)&d_B, bytes));

    dim3 block_size(32, 32);
    dim3 grid_size((n * n * n + block_size.x * block_size.y - 1) / (block_size.x * block_size.y)); // 2 grids

    gpuErrchk(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    kernel<<<grid_size, block_size>>>(d_A, d_B, n);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_dA, d_A, bytes, cudaMemcpyDeviceToHost));
    double end_time = getTimeStamp();

    computeOnCPU(h_A, h_B, n);

    // int i, j, k;
    // for (k = 0; k < n; k++)
    // {
    //     for (i = 0; i < n; i++)
    //     {
    //         for (j = 0; j < n; j++)
    //         {
    //             printf("h_dA: %.2f ", h_dA[k * n * n + i * n + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // for (k = 0; k < n; k++)
    // {
    //     for (i = 0; i < n; i++)
    //     {
    //         for (j = 0; j < n; j++)
    //         {
    //             printf("h_A: %.2f ", h_A[k * n * n + i * n + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    if (!checkEqual(h_A, h_dA, n))
    {
        printf("Error: the two matrices are not equal \n");
    }
    double sum = getSum(h_dA, n);
    int total_time_ms = (int)ceil((end_time - start_time) * 1000);
    printf("%lf %d\n", sum, total_time_ms);

    // free memory resources
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFreeHost(h_A));
    gpuErrchk(cudaFreeHost(h_B));
    gpuErrchk(cudaDeviceReset());
    return 0;
}