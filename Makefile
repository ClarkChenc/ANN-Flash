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
	@data=$(word 2, $(MAKECMDGOALS)) ; \
	algo=$(word 3, $(MAKECMDGOALS)) ; \
	thread_num=$(word 4, $(MAKECMDGOALS)) ; \
	topk=$(word 5, $(MAKECMDGOALS)) ; \
	: ${data:=$(DEFAULT_DATA)} ; \
	: ${algo:=$(DEFAULT_ALGO)} ; \
	: ${thread_num:=$(DEFAULT_THREAD_NUM)} ; \
	: ${topk:=$(DEFAULT_TOPK)} ; \
	echo "Running with algo=$$algo and data=$$data thread_num=$$thread_num topk=$$topk" ; \
	cd bin && ./main $$data $$algo $$thread_num $$topk;

rerun: build run

%:
	@: