/*
 * Skeleton code for use with Computer Architecture assignment 3, LIACS,
 * Leiden University.
 */

#ifndef __TIMING_H__
#define __TIMING_H__

#include <time.h>

int    timespec_subtract  (struct timespec *result,
                           struct timespec *x,
                           struct timespec *y);
void   print_elapsed_time (const char      *description,
                           struct timespec *end_time,
                           struct timespec *start_time);

static inline void
get_time(struct timespec *t)
{
#ifdef ENABLE_TIMING
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, t);
#endif
}

#endif /* ! __TIMING_H__ */
