/* STREAM + SM
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

#define BSIZE 1024

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
    // return 1;
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
                // B[i * n * n + j * n + k] = i * n * n + j * n + k;
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
                // printf("%d : %f %f %f %f %f %f \n", i * n * n + j * n + k, a, b, c, d, e, f);
            }
            // printf("\n");
        }
        // printf("\n");
    }
}

__global__ void kernel(float *A, float *B, int n, int noElems, int offset)
{

    __shared__ float sdata[BSIZE];
    int idx = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
    // printf("idx %d\n", idx);1
    int tid = idx + offset;
    int i = tid / (n * n);
    int z = tid % (n * n);
    int j = z / n;
    int k = z % n;
    int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (local_idx < BSIZE)
    {
        sdata[local_idx] = B[tid];
    }
    __syncthreads();

    // printf("l_idx: %d, idx: %d, sdata[idx]: %f \n", local_idx, idx, sdata[idx]);

    if (i < n && j < n && k < n && idx < noElems)
    {
        // printf("K: %d j: %d i: %d idx: %d\n", k, j, i, idx);
        float a = (i - 1 >= 0) ? B[(i - 1) * n * n + j * n + k] : 0;
        float b = (i + 1 < n) ? B[(i + 1) * n * n + j * n + k] : 0;

        float c, d, e, f;
        if (j == 0)
        {
            c = 0;
        }
        else if (local_idx < n)
        {
            c = B[i * n * n + (j - 1) * n + k];
        }
        else
        {
            c = sdata[local_idx - n];
        }
        if (j == n - 1)
        {
            d = 0;
        }
        else if (BSIZE - n <= local_idx && local_idx < BSIZE)
        {
            d = B[i * n * n + (j + 1) * n + k];
        }
        else
        {
            d = sdata[local_idx + n];
        }
        if (k == 0)
        {
            e = 0;
        }
        else if (local_idx == 0)
        {
            e = B[i * n * n + j * n + k - 1];
        }
        else
        {
            e = sdata[local_idx - 1];
        }
        if (k == n - 1)
        {
            f = 0;
        }
        else if (local_idx == BSIZE - 1)
        {
            f = B[i * n * n + j * n + k + 1];
        }
        else
        {
            f = sdata[local_idx + 1];
        }

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
                    // printf("da: %f ", h_dA[i * n * n + j * n + k]);
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
    size_t noElems = ((size_t)n) * n * n;
    size_t bytes = noElems * sizeof(float);

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

    // stream
    int n_stream = 4;
    cudaStream_t *stream = (cudaStream_t *)malloc(n_stream * sizeof(cudaStream_t));
    int noElemPS = noElems / n_stream;

    dim3 block_size(32, 32);

    double t_G_t_1 = getTimeStamp();
    for (int i = 0; i < n_stream - 1; i++)
    {
        gpuErrchk(cudaStreamCreate(&stream[i]));

        int offset = i * noElemPS;
        int bytesPerStream = (noElemPS + n * n) * sizeof(float);
        // printf("offset: %d, noE:%d\n", offset, bytesPerStream / sizeof(float));
        gpuErrchk(cudaMemcpyAsync(&d_B[offset], &h_B[offset], bytesPerStream, cudaMemcpyHostToDevice, stream[i]));

        dim3 grid_size((noElemPS + BSIZE - 1) / BSIZE);
        kernel<<<grid_size, block_size, 0, stream[i]>>>(d_A, d_B, n, noElemPS, offset);

        bytesPerStream = noElemPS * sizeof(float);
        gpuErrchk(cudaMemcpyAsync(&h_dA[offset], &d_A[offset], bytesPerStream, cudaMemcpyDeviceToHost, stream[i]));
    }
    int stream_idx = n_stream - 1;
    gpuErrchk(cudaStreamCreate(&stream[stream_idx]));
    int offset = stream_idx * noElemPS;
    int bytesPerStream = bytes - stream_idx * noElemPS * sizeof(float);
    noElemPS = bytesPerStream / sizeof(float);
    gpuErrchk(cudaMemcpyAsync(&d_B[offset], &h_B[offset], bytesPerStream, cudaMemcpyHostToDevice, stream[stream_idx]));
    // printf("offset: %d, noE:%d\n", offset, bytesPerStream / sizeof(float));
    dim3 grid_size((noElemPS + BSIZE - 1) / BSIZE);
    kernel<<<grid_size, block_size, 0, stream[stream_idx]>>>(d_A, d_B, n, noElemPS, offset);

    gpuErrchk(cudaMemcpyAsync(&h_dA[offset], &d_A[offset], bytesPerStream, cudaMemcpyDeviceToHost, stream[stream_idx]));

    for (int i = 0; i < n_stream; i++)
    {
        gpuErrchk(cudaStreamSynchronize(stream[i]));
    }
    double end_time = getTimeStamp();

    computeOnCPU(h_A, h_B, n);

    // int i, j, k;
    // for (k = 0; k < n; k++)
    // {
    //     for (i = 0; i < n; i++)
    //     {
    //         for (j = 0; j < n; j++)
    //         {
    //             printf("h_dA: %f ", h_dA[k * n * n + i * n + j]);
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
    //             printf("h_A: %f ", h_A[k * n * n + i * n + j]);
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
    for (int i = 0; i < n_stream; i++)
    {
        gpuErrchk(cudaStreamDestroy(stream[i]));
    }
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFreeHost(h_A));
    gpuErrchk(cudaFreeHost(h_B));
    gpuErrchk(cudaDeviceReset());
    return 0;
}