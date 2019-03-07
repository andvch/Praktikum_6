SHELL := /bin/bash
#MAKEFLAGS += --silent
N := 25 26 27 28
K := 01 13
I := 01 02 04 08

all: main generate print equals
main: main.cpp
	mpicxx $^ -o $@
generate: generate.cpp
	mpicxx $^ -o $@
.cpp:
	g++ $^ -o $@
submit: main
	for n in $(N); do \
		for k in $(K) $$n; do \
			for i in $(I); do \
				mpisubmit.pl -p $$i -w 00:05 --stdout out_$$k-$$n-$$i --stderr err_$$k-$$n-$$i ./main -- $$n $$k ; \
			done \
		done \
	done
.PHONY: report
report:
	sed -n "s/~//p" out_* > report
clean:
	rm main generate print equals out_* err_*
