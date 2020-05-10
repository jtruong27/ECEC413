.PHONY = default clean

SOURCE = gauss_eliminate.c

EXEC = gauss_eliminate

default: $(SOURCE)
	gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -fopenmp -std=c99 -Wall -O3 -lm
clean: $(ECEC)
	rm -rf $(EXEC)

