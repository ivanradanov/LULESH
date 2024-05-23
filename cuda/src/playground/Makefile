SOURCES = cublasTest.cu
OBJECTS = $(SOURCES:.cu=.o)
EXECUTABLES = $(SOURCES:.cu=)

RUNTIME_PATH=/home/twang/sources/projects/llvm-project2/memred-runtimes/trace.cpp
MEMRED_MEMORY_ANALYSIS_OUT=./.memred.memory.analysis.out

CUDA_PATH=/usr/local/cuda-12.3
NVCC=/home/twang/sources/projects/llvm-project2/build/bin/clang++
NVCC_FLAGS= -g -O3 -I. --cuda-path=$(CUDA_PATH) -mllvm --memred-mode=trace

LD=/home/twang/sources/projects/llvm-project2/build/bin/clang++
LDFLAGS = -g -O0 -I$(CUDA_PATH)/include --cuda-path=$(CUDA_PATH) -L$(CUDA_PATH)/lib64 -lcudart_static -ldl -lpthread -lrt $(RUNTIME_PATH)

all: cublasTest

%.o: %.cu
	@echo "Building $<"
	$(NVCC) -c $(NVCC_FLAGS) -o $@  $<

cublasTest: cublasTest.o
	@echo "Linking $<"
	$(LD) $^ $(LDFLAGS) -lcublas -lm -o $@

clean:
	rm -f *.o *~ $(OBJECTS) $(EXECUTABLES)
	rm -rf *.dSYM
	rm $(MEMRED_MEMORY_ANALYSIS_OUT)
