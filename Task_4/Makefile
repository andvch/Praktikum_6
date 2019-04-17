SHELL := /bin/bash
#MAKEFLAGS += --silent

all: main generate print equals
.cpp:
	mpicxx -std=c++11 -Wall -Werror -fopenmp $^ -o $@
.PHONY: lint
lint:
	cpplint *.cpp *.h
.PHONY: test
test: main equals
	for way in ./tests/*; do \
		echo -n -e "$$way \t" ; \
		./main $$way/A.dat $$way/C.dat < $$way/in.txt > /dev/null; \
		x=`./equals $$way/B.dat $$way/C.dat` ; \
		rm $$way/C.dat ; \
		if [[ $$x != 1 ]]; then \
			echo "FAIL" ; \
			exit 1 ; \
		else \
			echo "OK" ; \
		fi \
	done ;
.PHONY: clean
clean:
	rm main generate print equals
