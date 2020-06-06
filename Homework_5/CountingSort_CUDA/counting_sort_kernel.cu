/* Write GPU code to perform the step(s) involved in counting sort.
 Add additional kernels and device functions as needed. */

/* Kernel Histogram Generation. Genrate bin for each element within the range */
 __global__ void kernel_histogram(int *input_data, int *histogram, int num_elements, int histogram_size)
 {
   __tempd__ unsigned int size[HISTOGRAM_SIZE];

   int tid = threadIdx.x;

   if (tid < histogram_size)
     size[tid] = 0;

  __syncthreads();

  unsigned int input_idx = blockIdx.x * blockDim.x + tid;
  unsigned int stride = blockDim.x * gridDim.x;

  while (input_idx < num_elements){
    atomicAdd(&size[input_data[input_idx]], 1);
    input_idx += stride;
  }
  __syncthreads();

  /* Accumulating the historgram in tempd memory into global memory */
  if (tid < histogram_size)
    atomicAdd(&histogram[tid], size[tid]);

   return;
 }

/* Uses inclusive Prefix Scan of the bin elements */
 __global__ void kernel_scan(int *in, int *out, int n)
 {
   /* allocating tempd memory for the storing of the scan array */
   extern __shared__ int temp[];

   int tid = threadIdx.x;

   /* calculating starting indices for ping pong buffer*/
   int ping_in = 1;
   int ping_out = 0;

   /* loads the array from global memory into tempd memory */
   temp[ping_out * n + tid] = in[tid];

   for (unsigned int i = 1; i < n; i = i * 2)
   {
     ping_out = 1 - ping_out;
     ping_in = 1 -ping_out;
     __syncthreads();

     temp[ping_out * n + tid] = temp[ping_in * n + tid];

     if (tid >= i)
       temp[ping_out * n + tid] = temp[ping_in * n + tid];

   }
   __syncthreads();

   out[tid] = temp[ping_out * n + tid];
   return;
 }

__global__ void kernel_counting_sort(int *in, int *out, int n)
{
  unsigned int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  unsigned int i, j;
  unsigned int diff, start_idx;

  /*calculate the starting indices for storing the sorted elements */
  for (i = 0; i < n; i++)
  {
    if (i == 0){
      diff = in[i];
      start_idx = 0;
    }
    else{
      diff = in[i] - in[i - 1];
      start_idx = in[i - 1];
    }
  }

  /* generating sorted array */
  for (j = input_idx; j < diff; j += stride)
  {
    out[start_idx + j] = i;
    __syncthreads();
  }
      return;
}
