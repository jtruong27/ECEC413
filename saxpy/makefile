.PHONY = default clean

SOURCE = saxpy.c
EXEC = saxpy

default: $(SOURCE)
	gcc -o $(EXEC) $(SOURCE) -O3 -Wall -std=c99 -lpthread -lm

clean: $(EXEC)
	rm -rf $(EXEC)
