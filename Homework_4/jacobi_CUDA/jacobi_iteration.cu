/* Host code for the Jacobi method of solving a system of linear equations
 * by iteration.

 * Build as follws: make clean && make

 * Author: Naga Kandasamy
 * Date modified: May 21, 2020
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */
/* #define DEBUG */

int main(int argc, char **argv)
{
	if (argc > 1) {
		printf("This program accepts no arguments\n");
		exit(EXIT_FAILURE);
	}

    matrix_t  A;                    /* N x N constant matrix */
		matrix_t  B;                    /* N x 1 b matrix */
		matrix_t reference_x;           /* Reference solution */
		matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

	/* Initialize the random number generator */
	srand(time(NULL));

	/* Generate diagonally dominant matrix */
  printf("\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
	A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL) {
        printf("Error creating matrix\n");
        exit(EXIT_FAILURE);
	}

  /* Create the other vectors */
  B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);
	reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
  gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

		struct timeval start, stop;

    /* Compute Jacobi solution on CPU */
		printf("\nPerforming Jacobi iteration on the CPU\n");
		gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B);
		gettimeofday(&stop, NULL);
		fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
									(stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, reference_x, B); /* Display statistics */

	/* Compute Jacobi solution on device. Solutions are returned in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf("\nPerforming Jacobi iteration on device\n");
		compute_on_device(A, gpu_naive_solution_x, gpu_opt_solution_x, B);
		printf("\nGPU Naive Solution:\n");
    display_jacobi_solution(A, gpu_naive_solution_x, B); /* Display statistics and solution */
		printf("\nGPU Optimize Solution:\n");
    display_jacobi_solution(A, gpu_opt_solution_x, B); /* Display statistics snd solution */

    free(A.elements);
		free(B.elements);
		free(reference_x.elements);
		free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);

    exit(EXIT_SUCCESS);
}


/* FIXME: Complete this function to perform Jacobi calculation on device */
void compute_on_device(const matrix_t A, matrix_t gpu_naive_sol_x,
                       matrix_t gpu_opt_sol_x, const matrix_t B)
{
	struct timeval start, stop;

	int done = 0;
	int num_iter = 0;

	double ssd, mse;
	double *device_ssd = NULL; /* Pointer to device address holding ssd */

	/* Allocate matrices to hold iteration values */
	matrix_t new_x_naive = allocate_matrix_on_host (MATRIX_SIZE, 1, 0);
	matrix_t new_x_opt = allocate_matrix_on_host (MATRIX_SIZE, 1, 0);

	/* initialize solution of x for GPU */
	for (int i = 0; i < A.num_rows; i++)
	{
		float e = B.elements[i];
		gpu_naive_sol_x.elements[i] = e;
		gpu_opt_sol_x.elements[i] = e;
	}

	/* Allocating space on device for matricies on the GPU */
	matrix_t device_A = allocate_matrix_on_device(A);
	check_CUDA_error("Allocating matrix A");
	matrix_t device_navive_sol_x = allocate_matrix_on_device(gpu_naive_sol_x);
	check_CUDA_error("Allocating matrix naive_sol_x");
	matrix_t device_opt_sol_x = allocate_matrix_on_device(gpu_opt_sol_x);
	check_CUDA_error("Allocating matrix opt_sol_x");
	matrix_t device_B = allocate_matrix_on_device(B);
	check_CUDA_error("Allocating matrix B");
	matrix_t device_new_x_naiveive = allocate_matrix_on_device(new_x_naive);
	check_CUDA_error("Allocating new_x_naive");
	matrix_t device_new_x_opt_opt = allocate_matrix_on_device(new_x_opt);

	/* Copying matricies A, B, and x solutions to GPU */
	copy_matrix_to_device(device_A, A);
	check_CUDA_error("Copying matrix A over to device");
	copy_matrix_to_device(device_B, B);
	check_CUDA_error("Copying matrix B over to device");
	copy_matrix_to_device(device_navive_sol_x, gpu_naive_sol_x);
	check_CUDA_error("Copying matrix naive_sol_x over to device");
	copy_matrix_to_device(device_opt_sol_x, gpu_opt_sol_x);
	check_CUDA_error("Copying matrix opt_sol_x over to device");

	/* Allocating space for the device ssd on the GPU */
	cudaMalloc ((void**) &device_ssd, sizeof(double));

	/* Allocating space for the lock and initializing  mutex/locks on the GPU */
	int *mutex_on_device = NULL;
	cudaMalloc ((void **) &mutex_on_device, sizeof(int));
	cudaMemset (mutex_on_device, 0, sizeof(int));

	/* Setting up the execution configuration for the naive kernel */
	dim3 threads(1, THREAD_BLOCK_SIZE, 1);
	dim3 grid(1, (A.num_rows + THREAD_BLOCK_SIZE - 1)/ THREAD_BLOCK_SIZE);
	printf ("Performing Jacobi Naive Solution:\n");
	gettimeofday(&start, NULL);

	while (!done) {
		cudaMemset (device_ssd, 0.0, sizeof(double));

		/* using jacboi iteration kernel naive */
		jacobi_iteration_kernel_naive<<<grid, threads>>>(device_A, device_navive_sol_x, device_new_x_naiveive, device_B, mutex_on_device, device_ssd);
		check_CUDA_error("KERNEL FAILURE: jacobi_iteration_kernel_naive\n");
		cudaDeviceSynchronize ();

		jacobi_update_x<<<grid,threads>>>(device_navive_sol_x, device_new_x_naiveive);
		check_CUDA_error("KERNEL FAILURE: jacobi_update_x");
		cudaDeviceSynchronize();

		/* Check for convergence and update the unknowns. */
		cudaMemcpy (&ssd, device_ssd, sizeof(double), cudaMemcpyDeviceToHost);
		num_iter++;
		mse = sqrt(ssd); /* Mean squared error. */

		if (mse <= THRESHOLD){
			done = 1;
			printf ("\nConvergence achieved after %d iterations \n", num_iter);
		}
	}
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time for GPU-Naive = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
										(stop.tv_usec - start.tv_usec)/(float)1000000));

		/* Jacobi optimized kernel */
		threads.x = threads.y = TILE_SIZE;
		grid.x = 1;
		grid.y = (gpu_opt_sol_x.num_rows + TILE_SIZE - 1)/TILE_SIZE;

		printf("\nPerforming Jacobi Optimized Solution:\n");
		gettimeofday(&start, NULL);
		done = 0;
		num_iter = 0;

	while (!done){
		cudaMemset (device_ssd, 0.0, sizeof(double));

		/* using jacboi iteration kernel optimized */
		jacobi_iteration_kernel_optimized<<<grid, threads>>>(device_A, device_opt_sol_x, device_new_x_opt_opt, device_B, mutex_on_device, device_ssd);
    check_CUDA_error("KERNEL FAILURE: jacobi_iteration_kernel_optimized\n");
    cudaDeviceSynchronize ();

		jacobi_update_x<<<grid,threads>>>(device_opt_sol_x, device_new_x_opt_opt);
    check_CUDA_error("KERNEL FAILURE: jacobi_update_x");
    cudaDeviceSynchronize();

    /* Check for convergence and update the unknowns. */
    cudaMemcpy (&ssd, device_ssd, sizeof(double), cudaMemcpyDeviceToHost);
    num_iter++;
    mse = sqrt(ssd); /* Mean squared error. */

    if (mse <= THRESHOLD){
			done = 1;
      printf ("\nConvergence achieved after %d iterations \n", num_iter);
		}
	}
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time for GPU-Optimized = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
										(stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Copying the solutions back from GPU */
	copy_matrix_from_device(gpu_naive_sol_x, device_navive_sol_x);
	check_CUDA_error("Copying matrix device_navive_sol_x from device");
	copy_matrix_from_device(gpu_opt_sol_x, device_opt_sol_x);
	check_CUDA_error("Copying matrix device_opt_sol_x from device");

	/* Freeing memory on GPU/ Clean up device memory */
	cudaFree(device_A.elements);
	cudaFree(device_B.elements);
	cudaFree(device_navive_sol_x.elements);
	cudaFree(device_opt_sol_x.elements);
	cudaFree(device_ssd);

	cudaFree(mutex_on_device);
	cudaFree(device_new_x_naiveive.elements);
	cudaFree(device_new_x_opt_opt.elements);

	free (new_x_naive.elements);
	free (new_x_opt.elements);

	return;
}

/* Allocating the matrix on the device of same size as  matrix M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

	M.elements = (float *)malloc(size * sizeof(float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0)
            M.elements[i] = 0;
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}

    return M;
}

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			printf("%f ", M.elements[i * M.num_columns + j]);
        }

        printf("\n");
	}

    printf("\n");
    return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    return;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows;
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL)
        return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);

	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}

        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    return M;
}
