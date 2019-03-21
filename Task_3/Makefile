SHELL := /bin/bash
#MAKEFLAGS += --silent
I := 01
N := 28
P := 01 02 04
T := 01 02 04 08
E := 0.01

all: main generate print equals
main: main.cpp
	mpicxx -O3 -fopenmp $^ -o $@
generate: generate.cpp
	mpicxx $^ -o $@
.cpp:
	g++ $^ -o $@
submit: main
	for n in $(N); do \
		for p in $(P); do \
			for t in $(T); do \
				for ((i=0; i < $(I); ++i)) ; do \
					mpisubmit.pl -p $$p -w 00:05 -t $$t \
					--stdout out_$$n-$$p-$$t-$$i --stderr err_$$n-$$p-$$t-$$i \
					./main -- $$n $(E) ; \
				done \
			done \
		done \
	done
.PHONY: report
report:
	sed -n "s/~//p" out_* > report
.PHONY: plot
plot:
	gnuplot script.gnu
.PHONY: clean
clean:
	rm main generate print equals out_* err_*
