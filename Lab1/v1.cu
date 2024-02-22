// pinned memepry of 3 zeros
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// time stamp function in seconds
#include <sys/time.h>
#include <math.h>
double getTimeStamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

__global__ void f_siggen(float *d_X, float *d_Y, float *d_Z, int n, int m)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // n: n_row, m: n_col
    if (i < n && j < m)
    {
        float a = (i > 0) ? d_X[(i - 1) * m + j] : 0;
        float x = d_X[i * m + j];
        float b = (i < n - 1) ? d_X[(i + 1) * m + j] : 0;

        float c = (j > 1) ? d_Y[i * m + (j - 2)] : 0;
        float d = (j > 0) ? d_Y[i * m + (j - 1)] : 0;
        float y = d_Y[i * m + j];

        d_Z[i * m + j] = a + x + b - c - d - y;
    }
}

__host__ void matrixSumHost(float *h_X, float *h_Y, float *h_Z, int n_row, int n_col)
{
    for (int i = 0; i < n_row; i++)
    {
        for (int j = 0; j < n_col; j++)
        {
            float a = (i > 0) ? h_X[(i - 1) * n_col + j] : 0;
            float x = h_X[i * n_col + j];
            float b = (i < n_row - 1) ? h_X[(i + 1) * n_col + j] : 0;
            float c = (j > 1) ? h_Y[i * n_col + (j - 2)] : 0;
            float d = (j > 0) ? h_Y[i * n_col + (j - 1)] : 0;
            float y = h_Y[i * n_col + j];
            h_Z[i * n_col + j] = a + x + b - d - c - y;
        }
    }
}

__host__ void iniData(float *h_X, float *h_Y, int n_row, int n_col)
{
    for (int i = 0; i < n_row; i++)
    {
        for (int j = 0; j < n_col; j++)
        {
            h_X[i * n_col + j] = (float)((i + j) % 100) / 2.0;
            h_Y[i * n_col + j] = (float)3.25 * ((i + j) % 100);
        }
    }
}

__host__ bool checkEqual(float *h_Z, float *h_dZ, int n_row, int n_col)
{
    float epsilon = 1e-6; // Adjust based on acceptable tolerance
    for (int i = 0; i < n_row; i++)
    {
        for (int j = 0; j < n_col; j++)
        {
            // printf("%f ", h_dZ[i * n_col + j]);
            if (fabsf(h_Z[i * n_col + j] - h_dZ[i * n_col + j]) > epsilon)
            {
                return false;
            }
        }
        // printf("\n");
    }
    return true;
}

int main(int argc, char *argv[])
{

    // set matrix size
    int n_row = 0;
    int n_col = 0;
    if (argc != 3)
    {
        printf("Error: The number of arguments is not 2");
    }
    else
    {
        n_row = atoi(argv[1]);
        n_col = atoi(argv[2]);
    }
    int noElems = n_row * n_col;
    int bytes = noElems * sizeof(float);
    cudaError_t status;
    dim3 block_size(32, 32);
    dim3 grid_size((n_col + block_size.x - 1) / block_size.x, (n_row + block_size.y - 1) / block_size.y);

    // alloc memory host-size
    float *h_X, *h_Y, *h_dZ;
    status = cudaHostAlloc((void **)&h_X, bytes, 0);
    if (status != cudaSuccess)
    {
        printf("Error: cudaHostAlloc for h_X failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    status = cudaHostAlloc((void **)&h_Y, bytes, 0);
    if (status != cudaSuccess)
    {
        printf("Error: cudaHostAlloc for h_Y failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    float *h_Z = (float *)malloc(bytes);
    status = cudaHostAlloc((void **)&h_dZ, bytes, 0);
    if (status != cudaSuccess)
    {
        printf("Error: cudaHostAlloc for h_dZ failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    // initialization
    iniData(h_X, h_Y, n_row, n_col);

    // alloc memory dev-size
    float *d_X, *d_Y, *d_Z;
    status = cudaMalloc((void **)&d_X, bytes);
    if (status != cudaSuccess)
    {
        printf("Error: cudaMalloc for h_X failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    status = cudaMalloc((void **)&d_Y, bytes);
    if (status != cudaSuccess)
    {
        printf("Error: cudaMalloc for h_Y failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    status = cudaMalloc((void **)&d_Z, bytes);
    if (status != cudaSuccess)
    {
        printf("Error: cudaMalloc for d_Z failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    // transfer data to dev
    double t_G_t_1 = getTimeStamp();
    double CG_t_t_1 = getTimeStamp();
    status = cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        printf("Error: cudaMemcpy h_X -> d_X failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    status = cudaMemcpy(d_Y, h_Y, bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        printf("Error: cudaMemcpy h_Y -> d_Y failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    double CG_t_t_2 = getTimeStamp();

    // invoke a kernel
    double k_t_1 = getTimeStamp();
    f_siggen<<<grid_size, block_size>>>(d_X, d_Y, d_Z, n_row, n_col);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        printf("Error: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    double k_t_2 = getTimeStamp();

    // copy data back
    double GC_t_t_1 = getTimeStamp();
    status = cudaMemcpy(h_dZ, d_Z, bytes, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        printf("Error: cudaMemcpy d_Z -> h_dZ failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    double GC_t_t_2 = getTimeStamp();
    double t_G_t_2 = getTimeStamp();

    // check result
    matrixSumHost(h_X, h_Y, h_Z, n_row, n_col);
    printf("the two matrices are equal: %d\n", checkEqual(h_Z, h_dZ, n_row, n_col));

    // output time and exit
    float k_t = k_t_2 - k_t_1;
    float t_G_t = t_G_t_2 - t_G_t_1;
    float CG_t_t = CG_t_t_2 - CG_t_t_1;
    float GC_t_t = GC_t_t_2 - GC_t_t_1;
    float Z_v = h_dZ[5 * n_col + 5];
    printf("%.6f %.6f %.6f %.6f %.6f\n", t_G_t, CG_t_t, k_t, GC_t_t, Z_v);

    // free memory resources
    // free memory resources
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);

    cudaFreeHost(h_X);
    cudaFreeHost(h_Y);
    free(h_Z);
    cudaFreeHost(h_dZ);
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
