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

__global__ void f_siggen(float *d_X, float *d_Y, float *d_Z, int n, int m, int noElems, int offset)
{
    int idx = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
    int i = (idx + offset) / m; // matrix row index
    int j = (idx + offset) % m; // matrix column index
    // n: n_row, m: n_col
    // printf("idx: %d i: %d, j:%d\n", idx, i, j);
    if (i < n && j < m && idx < noElems)
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

void iniData(float *h_X, float *h_Y, int n_row, int n_col)
{
    for (int i = 0; i < n_row; i++)
    {
        for (int j = 0; j < n_col; j++)
        {
            h_X[i * n_col + j] = (float)((i + j) % 100) / 2.0;
            h_Y[i * n_col + j] = (float)3.25 * ((i + j) % 100);
        }
    }
    // for (int i = 0; i < n_row; i++)
    // {
    //     for (int j = 0; j < n_col; j++)
    //     {
    //         printf("X: %f ", h_X[i * n_col + j]);
    //     }
    //     printf("\n");
    // }
    // for (int i = 0; i < n_row; i++)
    // {
    //     for (int j = 0; j < n_col; j++)
    //     {
    //         printf("Y: %f ", h_Y[i * n_col + j]);
    //     }
    //     printf("\n");
    // }
}

__host__ bool checkEqual(float *h_Z, float *h_dZ, int n_row, int n_col)
{
    float epsilon = 1e-6;
    for (int i = 0; i < n_row; i++)
    {
        for (int j = 0; j < n_col; j++)
        {
            // printf("Z: %f ", h_dZ[i * n_col + j]);
            if (fabsf(h_Z[i * n_col + j] - h_dZ[i * n_col + j]) > epsilon)
            {
                return false;
            }
        }
        // printf("\n");
    }
    return true;
}

double CG_t_t_1 = 0;
double CG_t_t_2 = 0;
double GC_t_t_1 = 0;
double GC_t_t_2 = 0;
double k_t_1 = 0;
double k_t_2 = 0;

double k_t = 0;
double CG_t_t = 0;
double GC_t_t = 0;

void myCallBack1(cudaStream_t stream, cudaError_t status, void *data)
{
    CG_t_t_2 = getTimeStamp();
    k_t_1 = getTimeStamp();
    CG_t_t = CG_t_t + (CG_t_t_2 - CG_t_t_1);
}

void myCallBack2(cudaStream_t stream, cudaError_t status, void *data)
{
    k_t_2 = getTimeStamp();
    GC_t_t_1 = getTimeStamp();
    k_t = k_t + (k_t_2 - k_t_1);
}
void myCallBack3(cudaStream_t stream, cudaError_t status, void *userData)
{
    GC_t_t_2 = getTimeStamp();
    CG_t_t_1 = getTimeStamp();
    GC_t_t = GC_t_t + (GC_t_t_2 - GC_t_t_1);
}

int main(int argc, char *argv[])
{

    // set matrix size
    int n_row = 0;
    int n_col = 0;
    if (argc != 3)
    {
        printf("Error: The number of arguments is not 2");
        return -1;
    }
    else
    {
        n_row = atoi(argv[1]);
        n_col = atoi(argv[2]);
    }
    int deviceID = -1;
    cudaError_t status;
    status = cudaGetDevice(&deviceID);
    if (status != cudaSuccess)
    {
        printf("Error: Failed to get CUDA device ID\n");
        return -1;
    }
    cudaSetDevice(deviceID);
    int noElems = n_row * n_col;
    unsigned int bytes = noElems * sizeof(float);

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

    // stream
    int n_stream = 4;
    cudaStream_t *stream = (cudaStream_t *)malloc(n_stream * sizeof(cudaStream_t));

    dim3 block_size(1024);

    int noElemPS = noElems / n_stream;
    double t_G_t_1 = getTimeStamp();
    CG_t_t_1 = getTimeStamp();
    for (int i = 1; i <= n_stream; i++)
    {
        cudaStreamCreate(&stream[i]);
        int offset = (i - 1) * noElemPS;
        int bytesPerStream = (noElemPS + n_col) * sizeof(float);
        // printf("offset: %d bytesPS: %d noElemPS: %d\n", offset, bytesPerStream, noElemPS);

        // copy data
        cudaMemcpyAsync(&d_X[offset], &h_X[offset], bytesPerStream, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_Y[offset], &h_Y[offset], bytesPerStream, cudaMemcpyHostToDevice, stream[i]);
        cudaStreamAddCallback(stream[i], myCallBack1, (void *)&i, 0);

        dim3 grid_size((noElemPS + block_size.x - 1) / (block_size.x));
        // start kernel
        f_siggen<<<grid_size, block_size, 0, stream[i]>>>(d_X, d_Y, d_Z, n_row, n_col, noElemPS, offset);
        bytesPerStream = noElemPS * sizeof(float);
        cudaStreamAddCallback(stream[i], myCallBack2, (void *)&i, 0);
        // copy data back
        cudaMemcpyAsync(&h_dZ[offset], &d_Z[offset], bytesPerStream, cudaMemcpyDeviceToHost, stream[i]);
        cudaStreamAddCallback(stream[i], myCallBack3, (void *)&i, 0);
    }
    cudaStreamCreate(&stream[n_stream]);
    int offset = (n_stream - 1) * noElemPS;
    int bytesPerStream = bytes - (n_stream - 1) * noElemPS * sizeof(float);
    noElemPS = bytesPerStream / sizeof(float);
    // printf("offset: %d bytesPS: %d noElemPS: %d\n", offset, bytesPerStream, noElemPS);

    // copy data
    cudaMemcpyAsync(&d_X[offset], &h_X[offset], bytesPerStream, cudaMemcpyHostToDevice, stream[n_stream]);
    cudaMemcpyAsync(&d_Y[offset], &h_Y[offset], bytesPerStream, cudaMemcpyHostToDevice, stream[n_stream]);
    cudaStreamAddCallback(stream[n_stream], myCallBack1, (void *)&n_stream, 0);

    dim3 f_grid_size(((noElems - (n_stream - 1) * noElemPS) + block_size.x * block_size.y - 1) / (block_size.x * block_size.y));
    // start kernel
    f_siggen<<<f_grid_size, block_size, 0, stream[n_stream]>>>(d_X, d_Y, d_Z, n_row, n_col, noElemPS, offset);
    cudaStreamAddCallback(stream[n_stream], myCallBack2, (void *)&n_stream, 0);

    // copy data back
    cudaMemcpyAsync(&h_dZ[offset], &d_Z[offset], bytesPerStream, cudaMemcpyDeviceToHost, stream[n_stream]);
    cudaStreamAddCallback(stream[n_stream], myCallBack3, (void *)&n_stream, 0);

    for (int i = 1; i <= n_stream; i++)
    {
        cudaStreamSynchronize(stream[i]);
    }
    double t_G_t_2 = getTimeStamp();
    // check result
    matrixSumHost(h_X, h_Y, h_Z, n_row, n_col);
    if (!checkEqual(h_Z, h_dZ, n_row, n_col))
    {
        printf("Error: the two matrices are not equal \n");
    }

    // output time and exit
    double t_G_t = t_G_t_2 - t_G_t_1;
    float z_v = 0.0f;
    if (n_row >= 6 && n_col >= 6)
    {
        unsigned int idx = 5 * n_col + 5;
        z_v = (float)h_dZ[idx];
    }
    printf("%.6f %.6f %.6f %.6f %.6f \n", t_G_t, CG_t_t, k_t, GC_t_t, z_v);

    // free memory resources
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);
    for (int i = 0; i <= n_stream; i++)
    {
        cudaStreamDestroy(stream[i]);
    }
    cudaFreeHost(h_X);
    cudaFreeHost(h_Y);
    free(h_Z);
    cudaFreeHost(h_dZ);
    cudaDeviceReset();
    return 0;
}
