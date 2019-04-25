SHELL := /bin/bash
#MAKEFLAGS += --silent
I := 01 02 04 08
J := 01 02 04

all: main generate print equals
.cpp: quantum.h
	mpicxx -std=c++11 -Wall -Werror -fopenmp $^ -o $@
.PHONY: lint
lint:
	cpplint *.cpp *.h
.PHONY: test
test: main equals
	for way in ./tests/*; do \
		echo -n -e "$$way \t" ; \
		for i in $(I); do \
			for j in $(J); do \
				OMP_NUM_THREADS=$$j mpirun -n $$i --oversubscribe \
					./main $$way/A.dat $$way/C.dat < $$way/in.txt \
					> /dev/null ; \
				x=`./equals $$way/B.dat $$way/C.dat` ; \
				if [[ $$x != 1 ]]; then \
					rm $$way/C.dat ; \
					echo "FAIL" ; \
					exit 1 ; \
				fi \
			done \
		done ; \
		rm $$way/C.dat ; \
		echo "OK" ; \
	done
.PHONY: clean
clean:
	rm main generate print equals
