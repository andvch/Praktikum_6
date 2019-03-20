SHELL := /bin/bash
#MAKEFLAGS += --silent
N := 28
I := 01 02 04
J := 01 02 04 08

all: main generate print equals
#main: main.cpp
#	 xlC_r -qsmp=omp $^ -o $@
main: main.cpp
	mpicxx -fopenmp $^ -o $@
generate: generate.cpp
	mpicxx $^ -o $@
.cpp:
	g++ $^ -o $@
submit: main
	for n in $(N); do \
		for i in $(I) $$n; do \
			for i in $(J); do \
				mpisubmit.pl -p $$i -w 00:05 -t $$j --stdout out_$$n-$$i-$$j --stderr err_$$n-$$i-$$j ./main -- $$n 0.01 ; \
			done \
		done \
	done
.PHONY: report
report:
	sed -n "s/~//p" out_* > report
clean:
	rm main generate print equals out_* err_*
