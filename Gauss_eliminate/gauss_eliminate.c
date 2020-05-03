/* Date of last update: April 22, 2020
*
* Student names(s): Minjae Park & John Truong
* Date: 05/02/2020
*
* Compile as follows:
* gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -lpthread -lm
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"


#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define NUM_THREADS 32                                  /* providing number of threads */

/* Structure data type of p_threads arguments */
typedef struct thread_args {
    int tid;
    Matrix *U;

} THREAD_ARGS;

/* Function prototypes */
Matrix allocate_matrix(int, int, int);
extern int compute_gold (float *, unsigned int);
void * gauss_compute_gold(void *);                      /* new compute gold function to handle barrier sync */
void gauss_eliminate_using_pthreads(Matrix);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);


/* Initilizing Pthread Barriers for Sync */
pthread_barrier_t divBarr;                       /* divsion barrier */
pthread_barrier_t elimBarr;                      /* elimination barrier */

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
    fprintf(stderr, "CPU run time = %0.3f s\n", (float)(stop.tv_sec - start.tv_sec\
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

    /* FIXME: Perform Gaussian elimination using pthreads.
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");
    gettimeofday (&stop, NULL);
    gauss_eliminate_using_pthreads(U_mt);
    gettimeofday (&stop, NULL);
    fprintf(stderr, "Pthreads CPU run time = %0.3f s\n", (float)(stop.tv_sec - start.tv_sec\
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

/* Function to execute the Gaussian Elimination for the Matrix U while using pthread*/
void * gauss_compute_gold (void *args)
{
        float *U;
        unsigned int num_elements;
        unsigned int i, j, k;

        THREAD_ARGS *m_thread = (THREAD_ARGS *) args;
        num_elements = m_thread->U->num_rows;
        U = m_thread->U->elements;
        int tid = m_thread->tid;

        for (k = 0; k < num_elements; k++)
        {
            for (j = (k + tid + 1); j < num_elements; j+= NUM_THREADS)
            { /* reducing the current row */
              if (U[num_elements * k + k] == 0) {
                printf ("Numericsl instability. The principal diagonal element is zero.\n");
                return 0;
              }
              /* Division Step */
              U[num_elements * k + j] = (float) (U[num_elements * k + j] / U[num_elements * k + k]);
            } pthread_barrier_wait(&divBarr);

            for (i = (k + tid + 1); i < num_elements; i+= NUM_THREADS)
            {
              for (j = (k + 1); j < num_elements; j++)
              { /* reducing the current row */
                /* EliminationStep */
                U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]);	/* Elimination step. */
              }
              U[num_elements * i + k] = 0;
            } pthread_barrier_wait(&elimBarr);
        }

        /* setting the principal diagonal entry for each thread in a row */
        for (k = 0 + tid; k < num_elements; k += NUM_THREADS)
        { /* Set the principal diagonal entry in U Matrix to be 1. */
          U[num_elements * k + k] = 1;
        }

        /* free allocated data structures */
        free ((void *) m_thread);
        /* terminating the matrix thread when it is called and returns NULL */
        pthread_exit (NULL);
}


/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U)
{
        THREAD_ARGS *thread_args;                                /* arguments for threads */
        /* data structure to store the threads */
        pthread_t *worker_thread = (pthread_t *) malloc(NUM_THREADS * sizeof(pthread_t));
        /* Barrier Initialized */
        pthread_barrier_init(&divBarr, 0, NUM_THREADS);
        pthread_barrier_init(&elimBarr, 0, NUM_THREADS);

        int i;

        for (i = 0; i < NUM_THREADS; i++)
        {
                thread_args = (THREAD_ARGS *) malloc(sizeof(THREAD_ARGS));
                thread_args->tid = i;
                thread_args->U = &U;
                /* Failure to Create a worker thread and exits program */
                /* Creates independent threads each of which will execute the function */
                /* Start new thread in calling process and wait till threads are complete before guass function continues. */
                /* If wait, run the risk of executing an exit which will terminate the process and all threads before completing */
                if ((pthread_create (&worker_thread[i], NULL, gauss_compute_gold, (void *) thread_args)) != 0) {
                   printf("\nFailed to Create a Worker Thread\n");
                   exit(EXIT_FAILURE);
                }
        }

        for (i = 0; i < NUM_THREADS; i++)
        {
            pthread_join(worker_thread[i], NULL);
        }

        /* destroy barrier references and release resources used by barrier*/
        pthread_barrier_destroy(&divBarr);
        pthread_barrier_destroy(&elimBarr);

        return;
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
