/* Host-side code to perform counting sort
 * Author: Naga Kandasamy
 * Date modified: May 27, 2020
 *
 * Compile as follows: make clean && make
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "counting_sort.h"
#include "counting_sort_kernel.cu"

/* Uncomment to spit out debug info */
// #define DEBUG

extern "C" int counting_sort_gold(int *, int *, int, int);
int rand_int(int, int);
void print_array(int *, int);
void print_min_and_max_in_array(int *, int);
void compute_on_device(int *, int *, int, int);
int check_if_sorted(int *, int);
int compare_results(int *, int *, int);
void check_for_error(const char *);

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: %s num-elements\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    struct timeval start, stop;
    int num_elements = atoi(argv[1]);
    int range = MAX_VALUE - MIN_VALUE;
    int *input_array, *sorted_array_reference, *sorted_array_d;

    /* Populate input array with random integers between [0, RANGE] */
    printf("Generating input array with %d elements in the range 0 to %d\n", num_elements, range);
    input_array = (int *)malloc(num_elements * sizeof(int));
    if (input_array == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; i++)
        input_array[i] = rand_int (MIN_VALUE, MAX_VALUE);

#ifdef DEBUG
    print_array(input_array, num_elements);
    print_min_and_max_in_array(input_array, num_elements);
#endif

    /* Sort elements in input array using reference implementation.
     * The result is placed in sorted_array_reference. */
    printf("\nSorting array on CPU\n");
    int status;
    sorted_array_reference = (int *)malloc(num_elements * sizeof(int));
    if (sorted_array_reference == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    gettimeofday(&start, NULL);
    memset(sorted_array_reference, 0, num_elements);
    status = counting_sort_gold(input_array, sorted_array_reference, num_elements, range);
    if (status == -1) {
        exit(EXIT_FAILURE);
    }
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time for CPU = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
    										(stop.tv_usec - start.tv_usec) / (float)1000000));

    status = check_if_sorted(sorted_array_reference, num_elements);
    if (status == -1) {
        printf("Error sorting the input array using the reference code\n");
        exit(EXIT_FAILURE);
    }

    printf("Counting sort was successful on the CPU\n");

#ifdef DEBUG
    print_array(sorted_array_reference, num_elements);
#endif

    /* FIXME: Write function to sort elements in the array in parallel fashion.
     * The result should be placed in sorted_array_mt. */
    printf("\nSorting array on GPU\n");
    sorted_array_d = (int *) malloc(num_elements * sizeof(int));
    if (sorted_array_d == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    memset(sorted_array_d, 0, num_elements);
    compute_on_device(input_array, sorted_array_d, num_elements, range);

    /* Check the two results for correctness */
    printf("\nComparing CPU and GPU results\n");
    status = compare_results(sorted_array_reference, sorted_array_d, num_elements);
    printf(sorted_array_reference);
    printf(sorted_array_d);
    printf(num_elements);
    printf(status);
    if (status == 0)
        printf("Test passed\n");
    else
        printf("Test failed\n");

    exit(EXIT_SUCCESS);
}


/* FIXME: Write the GPU implementation of counting sort */
void compute_on_device(int *input_array, int *sorted_array, int num_elements, int range)
{
  int *device_input = NULL;
  int *device_output = NULL;

  /* initializing historgram step variables */
  int *device_hist = NULL;

  int *histogram = (int *) malloc(sizeof(int) * HISTOGRAM_SIZE);
  memset(histogram, 0, HISTOGRAM_SIZE);


  /* initializing prefix scan variables */
  int *device_scan = NULL;

  int *scan_out = (int *) malloc(sizeof(int) * HISTOGRAM_SIZE);
  memset(scan_out, 0 , sizeof(int) * HISTOGRAM_SIZE);



  /* Allocating space on GPU */
  cudaMalloc((void**) &device_input, num_elements * sizeof(int));
  cudaMalloc((void**) &device_output, num_elements * sizeof(int));
  cudaMalloc((void**) &device_hist, HISTOGRAM_SIZE * sizeof(int));
  cudaMalloc((void**) &device_scan, HISTOGRAM_SIZE * sizeof(int));


  /* CUDA copying memory from host to device */
  cudaMemcpy(device_output, sorted_array, num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_input, input_array, num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_hist, histogram, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_scan, scan_out, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyHostToDevice);

  /* Kernel Configuration */
  dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);
  dim3 grid(NUM_BLOCKS, 1);

  struct timeval start, stop;

  gettimeofday(&start, NULL);
  /* Histogram generation */
  kernel_histogram<<<grid, thread_block>>>(device_input, device_hist, num_elements, HISTOGRAM_SIZE);
  cudaDeviceSynchronize();
  check_for_error("Kernel Failure: Histogram");

  /* Kernel Configuration for Scanning */
  grid.x = 1;
  thread_block.x = HISTOGRAM_SIZE;

  int shared_memory_size = sizeof(int) * HISTOGRAM_SIZE;

  kernel_scan<<<grid, thread_block, 2 * shared_memory_size>>>(device_scan, device_hist, HISTOGRAM_SIZE);
  cudaDeviceSynchronize();
  check_for_error("Kernel Failure: Prefix Scanning");

  /* Kernel Configuration for Count Sort */
  grid.x = NUM_BLOCKS;
  thread_block.x  = THREAD_BLOCK_SIZE;

  kernel_counting_sort<<<grid, thread_block>>>(device_output, device_scan, HISTOGRAM_SIZE);
  cudaDeviceSynchronize();
  check_for_error("Kernel Failure: Counting Sort");
  gettimeofday(&stop, NULL);

  /* Copying results back from GPU */
  cudaMemcpy(sorted_array, device_output, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

  fprintf(stderr, "Execution time for GPU = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                      (stop.tv_usec - start.tv_usec) / (float)1000000));

  /* Freeing memory on GPU/ Clean up device memory */
  cudaFree(device_input);
  cudaFree(device_output);
  cudaFree(device_hist);
  cudaFree(device_scan);

  free(histogram);
  free(scan_out);

  return;
}

/* Check for errors for CUDA run time during Kernel execution */
void check_for_error(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err){
    printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return;
}

/* Check if array is sorted */
int check_if_sorted(int *array, int num_elements)
{
    int status = 0;
    int i;
    for (i = 1; i < num_elements; i++) {
        if (array[i - 1] > array[i]) {
            status = -1;
            break;
        }
    }

    return status;
}

/* Check if the arrays elements are identical */
int compare_results(int *array_1, int *array_2, int num_elements)
{
    int status = 0;
    int i;
    for (i = 0; i < num_elements; i++) {
        if (array_1[i] != array_2[i]) {
            status = -1;
            break;
        }
    }

    return status;
}

/* Return random integer between [min, max] */
int rand_int(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
    return (int)floorf(min + (max - min) * r);
}

/* Print given array */
void print_array(int *this_array, int num_elements)
{
    printf("Array: ");
    int i;
    for (i = 0; i < num_elements; i++)
        printf("%d ", this_array[i]);

    printf("\n");
    return;
}

/* Return min and max values in given array */
void print_min_and_max_in_array(int *this_array, int num_elements)
{
    int i;

    int current_min = INT_MAX;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] < current_min)
            current_min = this_array[i];

    int current_max = INT_MIN;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] > current_max)
            current_max = this_array[i];

    printf("Minimum value in the array = %d\n", current_min);
    printf("Maximum value in the array = %d\n", current_max);
    return;
}
