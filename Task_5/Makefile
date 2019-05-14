SHELL := /bin/bash
#MAKEFLAGS += --silent
N := 28
I := 01 02 04
J := 01 02 04 08

all: main generate print equals
main: main.cpp
	mpicxx -std=c++11 -Wall -Werror -fopenmp $^ -o $@
generate print equals:
	make -C ../Task_4 $@ && mv -t . ../Task_4/$@
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
					./main 0 $$way/A.dat $$way/C.dat > /dev/null ; \
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
.PHONY: submit
submit: main
	for n in $(N); do \
		for i in $(I); do \
			for j in $(J); do \
				mpisubmit.pl -p $$i -w 00:05 -t $$j \
				--stdout out_$$n-$$i-$$j --stderr err_$$n-$$i-$$j \
				./main $$n; \
			done \
		done \
	done
.PHONY: report
report:
	sed -n "s/~//p" out_* > report
.PHONY: clean
clean:
	rm main generate print equals out_* err_*
