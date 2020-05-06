.PHONY = default, clean, debug

SOURCES = jacobi_solver.c jacobi_solver.h compute_gold.c

default: $(SOURCES)
	gcc -o jacobi_solver $(SOURCES) -Wall -O3 -lpthread -lm

debug: $(SOURCES)
	gcc -o jacobi_solver $(SOURCES) -Wall -O3 -lpthread -lm -D DEBUG

clean: jacobi_solver
	rm -f jacobi_solver
