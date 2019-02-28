SHELL := /bin/bash
#MAKEFLAGS += --silent
n := 4

all: main generate print equals
main: main.cpp
	mpicxx $^ -o $@ -O3
generate: generate.cpp
	mpicxx $^ -o $@ -O3
.cpp:
	g++ $^ -o $@
%.dat: generate
	./generate $(n) $@
submit: main
	for n in 24 28 30; do \
		make n=$$n v_$$n.dat; \
		for k in 01 13 $$n; do \
			for i in 01 02 04 08; do \
				mpisubmit.bg -p $$i -w 00:05 --stdout out_$$k-$$n-$$i --stderr err_$$k-$$n-$$i ./main -- $$k v_$$n.dat ; \
			done \
		done \
	done
.PHONY: report
report:
	sed -n "s/~//p" out_* > report
clean:
	rm main generate print equals out_* err_* v_*.dat 
