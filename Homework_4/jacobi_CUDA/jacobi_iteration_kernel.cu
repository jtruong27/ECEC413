#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */

/* function is used to compare and swap technique to get a mutex/lock */
__device__ void lock(int *mutex)
{
  while(atomicCAS(mutex, 0, 1) != 0);
  return;
}

/* function uses atomic exchange operation to release the mutex/lock */
__device__ void unlock(int *mutex)
{
  atomicExch(mutex, 0);
  return;
}

/* Jacobi iteration using global and shared memory.Threads maintain good reference patterns to global memory via coalesced accesses */
__global__ void jacobi_update_x (matrix_t sol_x, const matrix_t new_x)
{
    unsigned int num_rows = sol_x.num_rows;

    /* Calculate thread index, block index and position in matrix */
    int threadY = threadIdx.y;
    int threadX = threadIdx.x;
    int blockY = blockIdx.y;
    int row = blockDim.y * blockY + threadY;

    if ((row < num_rows) && (threadX == 0)){
      sol_x.elements[row] = new_x.elements[row];
    }
    return;
}

/* Jacobi iteration using global memory. The reference pattern to global memory by the threads are not coalesced*/
__global__ void jacobi_iteration_kernel_naive(const matrix_t A, const matrix_t x, matrix_t x_update, const matrix_t B, int* mutex, double* ssd)
{
	__shared__ double ssd_per_thread[THREAD_BLOCK_SIZE];
  unsigned int num_rows = A.num_rows;
	unsigned int num_cols = A.num_columns;
  float new_x;
	double new_SSD = 0.0;

  /* Calculate thread index, block index and position in matrix */
  int threadY = threadIdx.y;
	int blockY = blockIdx.y;

  /* Obtain row number or position row in matrix */
  int row = blockDim.y * blockY + threadY;

  unsigned int i, j;

  if (row < num_rows){
    /* initilize jacobi sum */
    double sum = -A.elements[row * num_cols + row] * x.elements[row];

    for (j = 0; j < num_cols; j++)
    {
      sum += (double) A.elements[row * num_cols + j] * x.elements[j];
    }

    /* Finding new unknown values */
    new_x = (B.elements[row] - sum) / A.elements[row * num_cols + row];
    __syncthreads();

      new_SSD = (double) (new_x - x.elements[row]) * (new_x - x.elements[row]);
      x_update.elements[row] = new_x;

      ssd_per_thread[threadY] = new_SSD;
      __syncthreads();

      /* SSD Reduction */
      i = blockDim.y / 2;
      while (i != 0){
        if (threadY < i)
          ssd_per_thread[threadY] += ssd_per_thread[threadY + i];
          __syncthreads();
          i /= 2;
      }

      if (threadY == 0){
        lock(mutex);
        *ssd += ssd_per_thread[0];
        unlock(mutex);
      }
  }
    return;
}


__global__ void jacobi_iteration_kernel_optimized(const matrix_t A, const matrix_t x, matrix_t x_update, const matrix_t B, int* mutex, double* ssd)
{
  /* Declare shared memory for the thread block */
  __shared__ float aTile[TILE_SIZE][TILE_SIZE];
	__shared__ float xTile[TILE_SIZE];
	__shared__ double ssd_per_thread[TILE_SIZE];

  unsigned int num_rows = A.num_rows;
  unsigned int num_cols = A.num_columns;
  float new_x;
  double new_SSD = 0.0;
  double sum = 0.0;

  /* Calculate thread index, block index and position in matrix */
  int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockY = blockIdx.y;

  /* Obtain row number or position row in matrix */
	int row = blockDim.y * blockY + threadY;

  unsigned int i, k;

  if (row < num_rows){
    for (i = 0; i < num_cols; i += TILE_SIZE){
      /* Tile size elements are being brought in for row of A into shared memory */
      aTile[threadY][threadX] = A.elements[row * num_cols + i + threadX];

      /* Tile size elements are being brought in for row of x and B into shared memory */
      if (threadY == 0)
				xTile[threadX] = x.elements[i + threadX];
      /* Barrier sync to ensure that shared memory has been populated */
      __syncthreads();

      /* computing jacobi partial sum on the current tile */
      if (threadX == 0){
        for (k = 0; k < TILE_SIZE; k+=1)
          sum += (double) aTile[threadY][k] * xTile[k];
			}
			__syncthreads();
		}

    if (threadX == 0) {
			float aDiag = A.elements[row * num_cols + row];
    	float xDiag = x.elements[row];
	    float bDiag = B.elements[row];

      sum += -aDiag * xDiag;
      new_x = (bDiag - sum) / aDiag;

      new_SSD = (double) (new_x - xDiag) * (new_x - xDiag);
      x_update.elements[row] = new_x;

      ssd_per_thread[threadY] = new_SSD;
      __syncthreads();

      /* SSD Reduction */
      i = blockDim.y / 2;
      while (i != 0){
        if (threadY < i)
          ssd_per_thread[threadY] += ssd_per_thread[threadY + i];
          __syncthreads();
          i /= 2;
      }

      if (threadY == 0){
        lock(mutex);
        *ssd += ssd_per_thread[0];
        unlock(mutex);
      }
    }
  }
  return;
}
