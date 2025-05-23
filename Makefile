MSG=update

algo ?= hnsw
data ?= siftsmall

clean:
	rm -fr build && rm -fr bin && rm -f output.bin

debug-build: clean
	mkdir -p bin && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j32

debug: debug-build
	cd bin && gdb main

build: clean
	mkdir -p bin && mkdir build && cd build && cmake .. && make -j32

run:
	@algo=$(word 2, $(MAKECMDGOALS)) ; \
	data=$(word 3, $(MAKECMDGOALS)) ; \
	topk=$(word 4, $(MAKECMDGOALS)) ; \
	: ${algo:=$(DEFAULT_ALGO)} ; \
	: ${data:=$(DEFAULT_DATA)} ; \
	: ${topk:=$(DEFAULT_TOPK)} ; \
	echo "Running with algo=$$algo and data=$$data topk=$$topk" ; \
	cd bin && ./main $$data $$algo $$topk;

%:
	@: