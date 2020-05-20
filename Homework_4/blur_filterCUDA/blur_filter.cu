/* Reference code implementing the box blur filter.

    Build and execute as follows:
        make clean && make
        ./blur_filter size

    Author: Naga Kandasamy
    Date created: May 3, 2019
    Date modified: May 12, 2020

    FIXME: Minjae Park & John Truong
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* #define DEBUG */

/* Include the kernel code */
#include "blur_filter_kernel.cu"

/* setting number of threwads and block size */
#define NUM_THREAD_BLOCKS 240
#define THREAD_BLOCK_SIZE 128

extern "C" void compute_gold(const image_t, image_t);
void compute_on_device(const image_t, image_t);
int check_results(const float *, const float *, int, float);
void print_image(const image_t);
image_t allocate_image_on_device (const image_t);
void copy_image_to_device(image_t, image_t);
void copy_image_from_device(image_t, image_t);
void check_CUDA_error(const char *);

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s size\n", argv[0]);
        fprintf(stderr, "size: Height of the image. The program assumes size x size image.\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate memory for the input and output images */
    int size = atoi(argv[1]);

    fprintf(stderr, "Creating %d x %d images\n", size, size);
    image_t in, out_gold, out_gpu;
    in.size = out_gold.size = out_gpu.size = size;
    in.element = (float *)malloc(sizeof(float) * size * size);
    out_gold.element = (float *)malloc(sizeof(float) * size * size);
    out_gpu.element = (float *)malloc(sizeof(float) * size * size);
    if ((in.element == NULL) || (out_gold.element == NULL) || (out_gpu.element == NULL)) {
        perror("Malloc");
        exit(EXIT_FAILURE);
    }

    /* Poplulate our image with random values between [-0.5 +0.5] */
    srand(time(NULL));
    int i;
    for (i = 0; i < size * size; i++)
        in.element[i] = rand()/(float)RAND_MAX -  0.5;

    struct timeval start, stop;

   /* Calculate the blur on the CPU. The result is stored in out_gold. */
    fprintf(stderr, "Calculating blur on the CPU\n");
    gettimeofday(&start, NULL);
    compute_gold(in, out_gold);
    gettimeofday(&stop, NULL);
    fprintf ("Execution time for CPU = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                  (stop.tv_usec - start.tv_usec)/(float)1000000));

#ifdef DEBUG
   print_image(in);
   print_image(out_gold);
#endif

   /* FIXME: Calculate the blur on the GPU. The result is stored in out_gpu. */
   fprintf(stderr, "Calculating blur on the GPU\n");
   compute_on_device(in, out_gpu);

   /* Check CPU and GPU results for correctness */
   fprintf(stderr, "Checking CPU and GPU results\n");
   int num_elements = out_gold.size * out_gold.size;
   float eps = 1e-6;    /* Do not change */
   int check;
   check = check_results(out_gold.element, out_gpu.element, num_elements, eps);
   if (check == 0)
       fprintf(stderr, "TEST PASSED\n");
   else
       fprintf(stderr, "TEST FAILED\n");

   /* Free data structures on the host */
   free((void *)in.element);
   free((void *)out_gold.element);
   free((void *)out_gpu.element);

    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to calculate the blur on the GPU */
void compute_on_device(const image_t in, image_t out)
{
	/* Allocate memory on device for image */
	image_t d_in = allocate_image_on_device(in);
	image_t d_out = allocate_image_on_device(out);

	/* Copy image to memory of device */
	copy_image_to_device(d_in, in);

	/* Set up execution grid on the GPU */
  int num_thread_blocks = NUM_THREAD_BLOCKS;
	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1); /* Set number of threads in the thread block */
  fprintf(stderr, "Setting up a (%d x %d) execution grid\n", num_thread_blocks, num_thread_blocks);
  dim3 grid(NUM_THREAD_BLOCKS,1);

  fprintf(stderr, "\nKernel uses only gobal memory\n");
  struct timeval start, stop;
	gettimeofday(&start, NULL);
  /* Launch kernel with multiple thread blocks. The kernel call is non-blocking. */
	blur_filter_kernel<<<grid, thread_block>>>(d_in.element, d_out.element, d_in.size);
	gettimeofday(&stop, NULL);
	fprintf ("Execution time for GPU = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

  /* check for errors */
  check_CUDA_error("Error in kernel");

	/* Copy image out back over */
	copy_image_from_device(out, d_out);

	/* Free memory on GPU */
	cudaFree(d_in.element);
	cudaFree(d_out.element);

    return;
}

/* Allocate image on device */
image_t allocate_image_on_device (const image_t img)
{
	image_t img_device = img;
	int size = img.size * img.size * sizeof(float);

	cudaMalloc((void**)&img_device.element, size);
  if(img_device.element == NULL){
    fprintf(stderr, "CudaMalloc error\n");
    exit(EXIT_FAILURE);
  }

	return img_device;
}

/* Copy image from host memory to device memory */
void copy_image_to_device(image_t img_device, const image_t img_host)
{
	int size = (img_host.size * img_host.size) * sizeof (float);
	cudaMemcpy(img_device.element, img_host.element, size, cudaMemcpyHostToDevice);
}

/* Copy image from device memory to host memory */
void copy_image_from_device(image_t img_host, image_t img_device)
{
	int size = (img_device.size * img_device.size) * sizeof (float);
	cudaMemcpy(img_host.element, img_device.element, size, cudaMemcpyDeviceToHost);
}

/* Check for errors during kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

/* Check correctness of results */
int check_results(const float *pix1, const float *pix2, int num_elements, float eps)
{
    int i;
    for (i = 0; i < num_elements; i++)
        if (fabsf((pix1[i] - pix2[i])/pix1[i]) > eps)
            return -1;

    return 0;
}

/* Print out the image contents */
void print_image(const image_t img)
{
    int i, j;
    float val;
    for (i = 0; i < img.size; i++) {
        for (j = 0; j < img.size; j++) {
            val = img.element[i * img.size + j];
            printf("%0.4f ", val);
        }
        printf("\n");
    }

    printf("\n");
}
