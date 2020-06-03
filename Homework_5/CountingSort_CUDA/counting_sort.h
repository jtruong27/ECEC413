#ifndef _COUNTING_SORT_H_
#define _COUNTING_SORT_H_

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "counting_sort.h"
#include "counting_sort_kernel.cu"

#define HISTOGRAM_SIZE 256      /* initialize historgram size */

#define RANGE 255               /* input array will have integer elements ranging from 0 to 255 range */
#define NUM_ELEMENTS 100000000  /* Number of input integers */

extern "C" int counting_sort_gold(int *, int *, int, int);
int rand_int(int, int);
void print_array(int *, int);
void print_min_and_max_in_array(int *, int);
void compute_on_device(int *, int *, int, int);
int check_if_sorted(int *, int);
int compare_results(int *, int *, int);
void check_for_error(const char *);

#endif
