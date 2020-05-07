/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.
 *
 * Author: Naga Kandasamy
 * Date modified: April 22, 2020
 * 
 * Members: Minjae Park, John Truong
 * Last modified: May 06, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm
*/

#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "jacobi_solver.h"

/* Uncomment the line below to spit out debug information */
/* #define DEBUG */

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    struct timeval start, stop;   

    matrix_t A;             /* N x N constant matrix */
    matrix_t B;             /* N x 1 b matrix */
    matrix_t reference_x;   /* Reference solution */
    matrix_t mt_solution_x; /* Solution computed by pthread code */

    /* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
    srand(time(NULL));
    A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
    if (A.elements == NULL)
    {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
    }

    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
    reference_x = allocate_matrix(matrix_size, 1, 0);
    mt_solution_x = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
    print_matrix(A);
    print_matrix(B);
    print_matrix(reference_x);
#endif

    /* Compute Jacobi solution using reference code */
    fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
   
#ifdef DEBUG
    print_matrix(A);
    print_matrix(B);
    print_matrix(reference_x);
#endif

    /* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads\n");
    gettimeofday(&start, NULL);
    compute_using_pthreads(A, mt_solution_x, B);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */

#ifdef DEBUG
    print_matrix(A);
    print_matrix(B);
    print_matrix(mt_solution_x);
#endif

    free(A.elements);
    free(B.elements);
    free(reference_x.elements);
    free(mt_solution_x.elements);

    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads. 
 * Result must be placed in mt_sol_x. */
void compute_using_pthreads(const matrix_t A, matrix_t mt_sol_x, const matrix_t B)
{
    /* data structure to store the threads */
    pthread_t *tids = malloc(NUM_THREADS * sizeof(pthread_t));
    targs_t *targs = malloc(NUM_THREADS * sizeof(targs_t));

    // calculate chunk size
    int chunk = A.num_rows / NUM_THREADS;

    int i, done, num_iter;
    double ssd;

    // setup jacobi problem
    for (i = 0; i < NUM_THREADS; i++)
    {
        targs[i].A = A;
        targs[i].B = B;
        targs[i].x = &mt_sol_x;
        targs[i].tid = i;
        targs[i].chunk = chunk;
        pthread_create(&tids[i], NULL, jacobi_setup, &targs[i]);
    }

    for (i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(tids[i], NULL);
    }

    // allocate n x 1 matrix to hold iteration values
    matrix_t new_x = allocate_matrix(A.num_rows, 1, 0);
    
    // lock for global variable ssd
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    // initialize barrier for sync
    pthread_barrier_t barr;
    pthread_barrier_init(&barr, NULL, NUM_THREADS + 1);

    // initialize global variables
    done = 0;
    num_iter = 0;

    // perform jacobi iteration
    while (!done)
    {
        ssd = 0;

        // create threads
        for (i = 0; i < NUM_THREADS; i++)
        {
            targs[i].ssd = &ssd;
            targs[i].barr = &barr;
            targs[i].mutex = &mutex;
            targs[i].new_x = &new_x;
            pthread_create(&tids[i], NULL, jacobi_thread, &targs[i]);
        }

        // force sync
        pthread_barrier_wait(&barr);

        for (i = 0; i < NUM_THREADS; i++)
        {
            pthread_join(tids[i], NULL);
        }

        num_iter++;
        printf("Iteration %d: %f\n", num_iter, sqrt(ssd));

        // check for convergence
        if (sqrt(ssd) <= THRESHOLD)
            done = 1;
    }

    // cleanup
    pthread_mutex_destroy(&mutex);
    pthread_barrier_destroy(&barr);
    free((void *)tids);
    free((void *)targs);
}

void *jacobi_setup(void *args)
{
    int i;
    int tid, chunk, start_idx, end_idx;
    matrix_t B, *x;

    // upack arguments
    targs_t *targs = (targs_t *) args;
    B = targs->B;
    x = targs->x;
    tid = targs->tid;
    chunk = targs->chunk;

    // setup bounds
    start_idx = tid * chunk;
    end_idx = start_idx + chunk;

    // initialize current jacobi solution
    for (i = start_idx; i < end_idx; i++)
        x->elements[i] = B.elements[i];

    pthread_exit(NULL);
}

void *jacobi_thread(void *args)
{
    int i, j;
    int tid, chunk, start_idx, end_idx;
    int num_cols;
    matrix_t A, B, *x, *new_x;
    double *ssd, sum, partial_ssd;

    // unpack arguments
    targs_t *targs = (targs_t *) args;
    A = targs->A;
    B = targs->B;
    x = targs->x;
    new_x = targs->new_x;
    chunk = targs->chunk;
    tid = targs->tid;
    ssd = targs->ssd;
    num_cols = A.num_columns;

    // setup bounds
    start_idx = tid * chunk;
    end_idx = start_idx + chunk;

    for (i = start_idx; i < end_idx; i++)
    {
        sum = -A.elements[i * num_cols + i] * x->elements[i];
        for (j = 0; j < num_cols; j++)
            sum += A.elements[i * num_cols + j] * x->elements[j];
        
        // update values for the unknowns for the current row
        new_x->elements[i] = (B.elements[i] - sum) / (A.elements[i * num_cols + i]);
    }

    // barrier sync
    pthread_barrier_wait(targs->barr);

    // accumulate local covergence
    partial_ssd = 0.0;
    for (i = start_idx; i < end_idx; i++) {
        partial_ssd += (x->elements[i] - new_x->elements[i]) * (x->elements[i] - new_x->elements[i]);
        x->elements[i] = new_x->elements[i];
    }

    // update global convergence
    pthread_mutex_lock(targs->mutex);
    *(ssd) += partial_ssd;
    pthread_mutex_unlock(targs->mutex);

    pthread_exit(NULL);
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

    M.elements = (float *)malloc(size * sizeof(float));
    for (i = 0; i < size; i++)
    {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }

    return M;
}

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
    for (i = 0; i < M.num_rows; i++)
    {
        for (j = 0; j < M.num_columns; j++)
        {
            fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }

        fprintf(stderr, "\n");
    }

    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand() / (float)RAND_MAX;
    return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
    float diag_element;
    float sum;
    for (i = 0; i < M.num_rows; i++)
    {
        sum = 0.0;
        diag_element = M.elements[i * M.num_rows + i];
        for (j = 0; j < M.num_columns; j++)
        {
            if (i != j)
                sum += abs(M.elements[i * M.num_rows + j]);
        }

        if (diag_element <= sum)
            return -1;
    }

    return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(int num_rows, int num_columns)
{
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
    fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
    for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);

    /* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
    for (i = 0; i < num_rows; i++)
    {
        row_sum = 0.0;
        for (j = 0; j < num_columns; j++)
        {
            row_sum += fabs(M.elements[i * M.num_rows + j]);
        }

        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
    }

    /* Check if matrix is diagonal dominant */
    if (check_if_diagonal_dominant(M) < 0)
    {
        free(M.elements);
        M.elements = NULL;
    }

    return M;
}
