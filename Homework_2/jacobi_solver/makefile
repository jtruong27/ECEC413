.PHONY = default, clean

SOURCES = jacobi_solver.c jacobi_solver.h compute_gold.c

default: $(SOURCES)
	gcc -o jacobi_solver $(SOURCES) -Wall -O3 -lpthread -lm

clean: jacobi_solver
	rm -f jacobi_solver
