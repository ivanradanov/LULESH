SHELL = /bin/sh
.SUFFIXES: .cu .o

LULESH_EXEC = lulesh2.0

SOURCES2.0 = allocator.cu   lulesh-comms.cu   lulesh-comms-gpu.cu   lulesh.cu
OBJECTS2.0 = $(SOURCES2.0:.cu=.o)

RUNTIME_PATH=/scr/ivan/src/memred/memred-runtimes/trace.cpp
MEMRED_MEMORY_ANALYSIS_OUT=./.memred.memory.analysis.out

CUDA_PATH=/usr/local/cuda-12.3
NVCC=clang++
#NVCC=nvcc
NVCC_FLAGS= -g -O3 -I. --cuda-path=$(CUDA_PATH) -mllvm --memred-mode=trace

#Default build suggestions with OpenMP for g++
LD=clang++
#LD=nvcc
LDFLAGS = -g -O0 -I$(CUDA_PATH)/include --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 -lcudart_static -ldl -lpthread -lrt $(RUNTIME_PATH)

.cu.o: %.cu
	@echo "Building $<"
	$(NVCC) -c $(NVCC_FLAGS) -o $@  $<

all: $(LULESH_EXEC)

lulesh2.0: $(OBJECTS2.0) $(RUNTIME_PATH)
	@echo "Linking"
	$(LD) $(OBJECTS2.0) $(LDFLAGS) -lm -o $@

clean:
	/bin/rm -f *.o *~ $(OBJECTS) $(LULESH_EXEC)
	/bin/rm -rf *.dSYM
	rm $(MEMRED_MEMORY_ANALYSIS_OUT)

tar: clean
	cd .. ; tar cvf lulesh-2.0.tar LULESH-2.0 ; mv lulesh-2.0.tar LULESH-2.0
