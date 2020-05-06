/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 22, 2020
 *
 * Members: Minjae Park, John Truong
 * Modified: May 06, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "jacobi_solver.h"

/* Uncomment the line below to spit out debug information */ 
#define DEBUG

/* declaring Pthread Barriers for Sync */
pthread_barrier_t barr;
/* declaring semaphore for protected writing */
typedef struct critical_section_s {
  pthread_mutex_t mutex;
  double value;
} critical_section_t;

critical_section_t ssd;
/* global exit flag */
int done;
int num_iter = 0;

int main(int argc, char **argv) 
{
  if (argc < 2) {
    fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
    fprintf(stderr, "matrix-size: width of the square matrix\n");
    exit(EXIT_FAILURE);
  }

  int matrix_size = atoi(argv[1]);

  matrix_t  A;                    /* N x N constant matrix */
  matrix_t  B;                    /* N x 1 b matrix */
  matrix_t reference_x;           /* Reference solution */ 
  matrix_t mt_solution_x;         /* Solution computed by pthread code */

  /* Generate diagonally dominant matrix */
  fprintf(stderr, "\nCreating input matrices\n");
  srand(time(NULL));
  A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
  if (A.elements == NULL) {
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
  compute_gold(A, reference_x, B, max_iter);
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
  compute_using_pthreads(A, mt_solution_x, B);
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
void compute_using_pthreads (const matrix_t A, matrix_t mt_sol_x, const matrix_t B)
{
  thread_args_t *thread_args = (thread_args_t *) malloc(sizeof(thread_args_t));

  /* data structure to store the threads */
  pthread_t *worker_thread = (pthread_t *) malloc(NUM_THREADS * sizeof(pthread_t));
  /* Barrier initialized */
  pthread_barrier_init(&barr, 0, NUM_THREADS);
  /* lock for global variable ssd */
  pthread_mutex_init(&ssd.mutex, NULL); /* initialize the mutex */

  int i;

  /* initialize global variables */
  done = 0;
  ssd.value = 0.0;

  for(i=0; i < NUM_THREADS; i++){
    thread_args->tid = i;
    thread_args->A = &A;
    thread_args->x = &mt_sol_x;
    thread_args->B = &B;
    /* Failure to Create a worker thread and exits program */
    /* Creates independent threads each of which will execute the function */
    /* Start new thread in calling process and wait till
     * threads are complete before guass function continues. */
    /* If wait, run the risk of executing an exit which will
     * terminate the process and all threads before completing */
    if ((pthread_create (&worker_thread[i], NULL, jacobi_thread, (void *) thread_args)) != 0) {
      printf("\nFailed to Create a Worker Thread\n");
      exit(EXIT_FAILURE);
    }
  }

  for (i = 0; i < NUM_THREADS; i++)
  {
    pthread_join(worker_thread[i], NULL);
  }

  pthread_barrier_destroy(&barr);
  pthread_mutex_destroy(&ssd.mutex);
  free((void *)worker_thread);
  free((void *)thread_args);

  return;
}

void *jacobi_thread(void *args){
  /* declare variables */
  matrix_t *A, *x, *B;
  int tid, r, c;
  /* loop variables */
  int i, j;
  /* barrier return value */
  int barr_ret;

  /* unpack arguments */
  thread_args_t *arguments = (thread_args_t *)args;
  tid = arguments->tid;
  A = arguments->A;
  B = arguments->B;
  x = arguments->x;
  r = A->num_rows;
  c = A->num_columns;

#ifdef DEBUG
  print_matrix(*A);
  print_matrix(*B);
  print_matrix(*x);
#endif

  /* Allocate n x 1 matrix to hold iteration values */
  matrix_t new_x = allocate_matrix(r, 1, 0);

  /* Initialize current jacobi solution */
  for (i = 0; i < r; i++)
    x->elements[i] = B->elements[i];

  /* perform jacobi iteration */
  double partial_ssd, sum;

  while(!done){
    /* reset accumulator */
    partial_ssd = 0.0;

    for (i = tid; i < r; i += NUM_THREADS){
      sum = -A->elements[i * c + i] * x->elements[i];
      for (j = 0; j < c; j++)
        sum += A->elements[i * c + j] * x->elements[i];

      /* Update values for the unknowns for the current row */
      new_x.elements[i] = (B->elements[i] - sum)/A->elements[i * c + i];

      /* add to convergence value */
      partial_ssd += (new_x.elements[i] - x->elements[i]) * (new_x.elements[i] - x->elements[i]);
    }

    /* update global convergence value */
    pthread_mutex_lock(&ssd.mutex);
    ssd.value += partial_ssd;
    pthread_mutex_unlock(&ssd.mutex);

    /* barrier synchronization */
    barr_ret = pthread_barrier_wait(&barr);

    /* check for convergence and update the unknowns */
    if(barr_ret == PTHREAD_BARRIER_SERIAL_THREAD) {
      num_iter++;
      double mse = sqrt(ssd.value);

      /* reset ssd */
      pthread_mutex_lock(&ssd.mutex);
      ssd.value = 0.0;
      pthread_mutex_unlock(&ssd.mutex);

      fprintf(stderr, "TID: %d. Iteration: %d. MSE = %f\n", tid, num_iter, mse);
      if (mse <= THRESHOLD)
        done = 1;
    }

    pthread_barrier_wait(&barr);

    /* update x */
    for(i = tid; i < r; i += NUM_THREADS)
      x->elements[i] = new_x.elements[i];

  }

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
  for (i = 0; i < size; i++) {
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
  for (i = 0; i < M.num_rows; i++) {
    for (j = 0; j < M.num_columns; j++) {
      fprintf(stderr, "%f ", M.elements[i * M.num_rows + j]);
    }

    fprintf(stderr, "\n");
  } 

  fprintf(stderr, "\n");
  return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
  float r = rand ()/(float)RAND_MAX;
  return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
  int i, j;
  float diag_element;
  float sum;
  for (i = 0; i < M.num_rows; i++) {
    sum = 0.0; 
    diag_element = M.elements[i * M.num_rows + i];
    for (j = 0; j < M.num_columns; j++) {
      if (i != j)
        sum += abs(M.elements[i * M.num_rows + j]);
    }

    if (diag_element <= sum)
      return -1;
  }

  return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix (int num_rows, int num_columns)
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
  for (i = 0; i < num_rows; i++) {
    row_sum = 0.0;		
    for (j = 0; j < num_columns; j++) {
      row_sum += fabs(M.elements[i * M.num_rows + j]);
    }

    M.elements[i * M.num_rows + i] = 0.5 + row_sum;
  }

  /* Check if matrix is diagonal dominant */
  if (check_if_diagonal_dominant(M) < 0) {
    free(M.elements);
    M.elements = NULL;
  }

  return M;
}



