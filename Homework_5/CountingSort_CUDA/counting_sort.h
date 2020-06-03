#ifndef _COUNTING_SORT_H_
#define _COUNTING_SORT_H_

/* Do not change the range value */
#define MIN_VALUE 0
#define MAX_VALUE 255

#define HISTOGRAM_SIZE 256      /* initialize historgram size */

#define RANGE 255               /* input array will have integer elements ranging from 0 to 255 range */
#define NUM_ELEMENTS 100000000  /* Number of input integers */

#define THREAD_BLOCK_SIZE 256   /* thread block size */
#define NUM_BLOCKS 40           /* initialize number of blocks */

#endif
