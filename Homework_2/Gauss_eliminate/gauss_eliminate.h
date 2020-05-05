#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

/* Matrix dimensions. */
#define NUM_COLUMNS MATRIX_SIZE     /* Number of columns */
#define NUM_ROWS MATRIX_SIZE        /* Number of rows */

/* Matrix Structure declaration */
typedef struct {
    unsigned int num_columns;   /* Width of the matrix */
    unsigned int num_rows;      /* Height of the matrix */
    float* elements;            /* Pointer to the first element of the matrix */
} Matrix;

#endif /* _MATRIXMUL_H_ */
