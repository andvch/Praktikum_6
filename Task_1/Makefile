SHELL := /bin/bash
#MAKEFLAGS += --silent

all: main
#main: main.cpp
#	 xlC_r -qsmp=omp $^ -o $@
.cpp:
	g++ -fopenmp $^ -o $@
submit: main
	for n in 20 24 28 32; do \
		for k in 01 13 $$n; do \
			for i in 1 2 4 8; do \
				bsub -W 10:00 -q normal -o out_$$k-$$n-$$i -e err_$$k-$$n-$$i OMP_NUM_THREADS=$$i ./main $$n $$k ; \
			done \
		done \
	done
.PHONY: report
report:
	sed -n "s/~//p" out_* > report
clean:
	rm main out_* err_*
