#
# Skeleton code for use with Computer Architecture assignment 3, LIACS,
# Leiden University.
#

CC = gcc
CFLAGS = -Wall -O3 -std=gnu99
LDFLAGS = -lm
PNGFLAGS = `pkg-config --cflags --libs libpng`

# Comment these lines to disable the timing code. (macOS does not
# provide clock_gettime().
CFLAGS += -DENABLE_TIMING
LDFLAGS += -lrt

NVCC = nvcc
NVCCFLAGS = -O3 -DENABLE_TIMING --use_fast_math

# Add targets for your other implementations here
TARGETS = \
	fotowinkel-base	\
	fotowinkel-MAP \
	fotowinkel-CR \
	fotowinkel-AC \
	fotowinkel-SIMD \
	fotowinkel-cuda 

COMMON_SRC = main.c image.c timing.c

all:			$(TARGETS)

fotowinkel-base:	main.c $(COMMON_SRC) filters-base.c pipeline.c
			$(CC) $(CFLAGS) -o $@ $^ $(PNGFLAGS) $(LDFLAGS)

# As exception we do not use a "pipeline.c" for the GPU version, but
# include "run-filters" in filters-cuda.cu.
fotowinkel-cuda:	main.c $(COMMON_SRC) filters-cuda.cu
			$(NVCC) $(NVCCFLAGS) -o $@ $^ $(PNGFLAGS) $(LDFLAGS)
			
fotowinkel-MAP:	main.c $(COMMON_SRC) filters-MAP.c pipeline.c
			$(CC) $(CFLAGS) -o $@ $^ $(PNGFLAGS) $(LDFLAGS)
			
fotowinkel-CR:	main.c $(COMMON_SRC) filters-CR.c pipeline-CR.c
			$(CC) $(CFLAGS) -o $@ $^ $(PNGFLAGS) $(LDFLAGS)

fotowinkel-AC:	main.c $(COMMON_SRC) filters-AC.c pipeline.c
			$(CC) $(CFLAGS) -o $@ $^ $(PNGFLAGS) $(LDFLAGS)

fotowinkel-SIMD: main.c $(COMMON_SRC) filters-SIMD.c pipeline.c
			$(CC) $(CFLAGS) -o $@ $^ $(PNGFLAGS) $(LDFLAGS)
# Add rules for your other implementations with new filters.c files here.


clean:
			rm -f $(TARGETS)
