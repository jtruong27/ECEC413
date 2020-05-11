.PHONY = default clean

SOURCE = jacobi_solver.c

EXEC = jacobi_solver

default: $(SOURCE)
	gcc -o jacobi_solver jacobi_solver.c compute_gold.c -fopenmp -std=c99 -Wall -O3 -lm

clean: $(ECEC)
	rm -rf $(EXEC) 
