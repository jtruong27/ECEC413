/* Gaussian elimination code.
 *
 * Author: Naga Kandasamy
 * Date of last update: April 29, 2020
 *
 * Student names(s): FIXME
 * Date: FIXME
 *
 * Compile as follows:
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -fopenmp -std=c99 -Wall -O3 -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "gauss_eliminate.h"
#include <omp.h>


/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_omp(Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);


int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);

    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);

    int status = compute_gold(U_reference.elements, A.num_rows);

    gettimeofday(&stop, NULL);
    fprintf(stderr, "Single Thread-CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }

    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");

    /* FIXME: Perform Gaussian elimination using OpenMP.
     * The resulting upper triangular matrix should be returned in U_mt */
    gettimeofday(&start, NULL);
    fprintf(stderr, "\nPerforming gaussian elimination using omp\n");
    gauss_eliminate_using_omp(U_mt);
    gettimeofday(&stop, NULL);

    fprintf(stderr, "OPM-CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using omp */
void gauss_eliminate_using_omp(Matrix U)
{
  unsigned int num_elements = MATRIX_SIZE;
  unsigned int i, j, k;

  printf("Starting Parallel Guassian Elimination OMP with %d elements.\n", num_elements);

  #pragma omp parallel num_threads(thread_count) shared(num_elements, U_mt) private(i, j, k)
  {
    #pragma omp for schedule(dynamic)
    for (k = 0; k < num_elements; k++)
    {
      for (j = (k + tid + 1); j < num_elements; j++)
        { /* reducing the current row */
          if (U_mt.elements[num_elements * k + k] == 0) {
            printf ("Numerical instability. The principal diagonal element is zero.\n");
            return 0;
          }
          /* Division Step */
          U_mt.elements[num_elements * k + j] = (float) (U_mt.elements[num_elements * k + j] / U_mt.elements[num_elements * k + k]);
        }

        /* Set the principal diagonal entry in U_mt to 1 */
        U_mt.elements[num_elements * k +k] = 1;

      for (i = (k + 1); i < num_elements; i++)
      {
        for (j = (k +1); j < num_elements; j++)
          /* Elimination Step */
          U_mt.elements[num_elements * i + j] = U_mt.elements[num_elements * i + j] -\
            (U_mt.elements[num_elements * i + k] * U_mt.elements[num_elements * k + j]);

			  U_mt.elements[num_elements * i + k] = 0;          
      }
    }
  }
}


/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.
 * If init == 1, perform random initialization.
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));

    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }

    return M;
}

/* Return a random floating-point number between [min, max] */
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;

    return 0;
}
