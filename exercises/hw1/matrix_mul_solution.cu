#include <stdio.h>

// these are just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg)                             \
  do                                                     \
  {                                                      \
    cudaError_t __err = cudaGetLastError();              \
    if (__err != cudaSuccess)                            \
    {                                                    \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
              msg, cudaGetErrorString(__err),            \
              __FILE__, __LINE__);                       \
      fprintf(stderr, "*** FAILED - ABORTING\n");        \
      exit(1);                                           \
    }                                                    \
  } while (0)

const int DSIZE = 8192;
const int block_size = 32; // CUDA maximum is 1024 *total* threads in block
const float A_val = 3.0f;
const float B_val = 2.0f;

// matrix multiply (naive) kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds)
{

  int idx = threadIdx.x + blockDim.x * blockIdx.x; // create thread x index
  int idy = threadIdx.y + blockDim.y * blockIdx.y; // create thread y index

  if ((idx < ds) && (idy < ds)) // Matrix dimensions if greater than 8192
  {
    float temp = 0;
    for (int i = 0; i < ds; i++)
      temp += A[idy * ds + i] * B[i * ds + idx]; // dot product of row and column

    // Grid Size = 256 x 256
    // For block 0 blockDim.x = 0
    // Block size = 32 x 32 = 1024 blockIdx.x = 0 to 255

    // Block 0,0 Thread 0,0
    //  A[0 x 8192 +  0   ]* B[0    x 8192 + 0 ]
    //  A[0 x 8192 +  1   ]* B[1    x 8192 + 0 ]
    //  A[0 x 8192 +  2   ]* B[2    x 8192 + 0 ]
    //  A[0 x 8192 + 8191 ]* B[8191 x 8192 + 0 ]

    // Block 0,0 Thread 0,1
    //  A[1 x 8192 + 0    ]* B[0    x 8192 + 0 ]
    //  A[1 x 8192 + 1    ]* B[1    x 8192 + 0 ]
    //  A[1 x 8192 + 2    ]* B[2    x 8192 + 0 ]
    //  A[1 x 8192 + 8191 ]* B[8191 x 8192 + 1 ]

    // Block 0,0 Thread 31,31
    //  A[31   x 8192 +    0 ]* B[0    x 8192 + 31 ]
    //  A[31   x 8192 +    1 ]* B[1    x 8192 + 31 ]
    //  A[31   x 8192 +    2 ]* B[2    x 8192 + 31 ]
    //  A[31   x 8192 + 8191 ]* B[8191 x 8192 + 31 ]

    // For block 1,1 Thread 0,1
    // A[32 x 8192 +   0 ]* B[0    x 8192 + 32 ]
    // A[32 x 8192 +   1 ]* B[1    x 8192 + 32 ]
    // A[32 x 8192 +   2 ]* B[2    x 8192 + 32 ]
    // A[32x8192   +8192 ]* B[8192 x 8192 + 32 ]

    // For block 255,255 Thread 31,31
    // A[8191 x 8192 +   0 ]* B[0    x 8192 + 8191 ]
    // A[8191 x 8192 +   1 ]* B[1    x 8192 + 8191 ]
    // A[8191 x 8192 +   2 ]* B[2    x 8192 + 8191 ]
    // A[8191 x 8192 +8191 ]* B[8191 x 8192 + 8191 ]

    C[idy * ds + idx] = temp;
  }
}

int main()
{

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;

  // start timing
  t0 = clock();

  h_A = new float[DSIZE * DSIZE];
  h_B = new float[DSIZE * DSIZE];
  h_C = new float[DSIZE * DSIZE];
  // Matrix size would be 8192 x 8192

  for (int i = 0; i < DSIZE * DSIZE; i++)
  {
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;
  }

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
  cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
  cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));

  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size); // dim3 variable holds 3 dimensions

  dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);

  // Block size would be 32 x 32 = 1024
  // So total grid size would be 256 x 256

  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < DSIZE * DSIZE; i++)
    if (h_C[i] != A_val * B_val * DSIZE)
    {
      printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val * B_val * DSIZE);
      return -1;
    }
  printf("Success!\n");
  return 0;
}
