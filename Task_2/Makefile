SHELL := /bin/bash
#MAKEFLAGS += --silent
N := 25 26 27 28
K := 01 13
I := 01 02 04 08
G := 08

all: main generate print equals
main: main.cpp
	mpicxx $^ -o $@
generate: generate.cpp
	mpicxx $^ -o $@
.cpp:
	g++ $^ -o $@
vec: generate
	for n in $(N); do \
		mpisubmit.pl -p $(G) -w 00:15 --stdout out_gen --stderr err_gen ./generate -- $$n v_$$n.dat ; \
	done
submit: main
	for n in $(N); do \
		for k in $(K) $$n; do \
			for i in $(I); do \
				mpisubmit.pl -p $$i -w 00:15 --stdout out_$$k-$$n-$$i --stderr err_$$k-$$n-$$i ./main -- $$k v_$$n.dat ; \
			done \
		done \
	done
.PHONY: report
report:
	sed -n "s/~//p" out_* > report
clean:
	rm main generate print equals out_* err_* v_*.dat 
