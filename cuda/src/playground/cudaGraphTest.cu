#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#define CUDA_CHECK(err)                                                  \
  do {                                                                   \
    cudaError_t err_ = (err);                                            \
    if (err_ != cudaSuccess) {                                           \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                            \
    }                                                                    \
  } while (0)

struct MemRedAnalysisParser {
  static constexpr std::string_view ANALYSIS_FILE = "./.memred.memory.analysis.out";

  struct KernelInfo {
    std::string funcName;
    std::string memoryEffect;
    std::vector<std::pair<int, std::string>> ptrArgInfos;
  };

  std::map<std::string, KernelInfo> funcNameToKernelInfoMap;

  MemRedAnalysisParser() {
    std::ifstream in(ANALYSIS_FILE.data());
    std::string ignore;
    while (in) {
      KernelInfo kernelInfo;

      // Read the function name
      // Example: "Function void CalcMinDtOneBlock<1024>(double*, double*, double*, double*, int) (@_Z17CalcMinDtOneBlockILi1024EEvPdS0_S0_S0_i):"
      char ch;
      do {
        in.read(&ch, 1);
      } while (in && ch != '@');
      if (!in) break;

      std::string funcName;
      in >> funcName;
      funcName.erase(funcName.size() - 2);
      kernelInfo.funcName = funcName;

      // Read the function's memory effect
      // Example: "Memory Effect: ArgMemOnly"
      std::string memoryEffect;
      in >> ignore >> ignore >> memoryEffect;
      kernelInfo.memoryEffect = memoryEffect;

      // Read the argument information
      // Example: "Arg #0:	Effect: ReadOnly  Capture: No"
      while (1) {
        std::string argumentKeyword;
        in >> argumentKeyword;
        if (argumentKeyword == "Function" || !in)
          break;
        if (argumentKeyword != "Arg") {
          abort();
        }

        in.read(&ch, 1);  // ' '
        in.read(&ch, 1);  // '#'

        size_t argumentIndex;
        in >> argumentIndex;

        std::string ptrArgEffect;

        // :  Effect: ReadOnly Capture: No
        in >> ignore >> ignore >> ptrArgEffect >> ignore >> ignore;

        kernelInfo.ptrArgInfos.push_back({argumentIndex, ptrArgEffect});
      }

      funcNameToKernelInfoMap[funcName] = kernelInfo;
    }
  }
};

void analyzeGraph(cudaGraph_t graph) {
  MemRedAnalysisParser analysisParser;

  size_t numNodes;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &numNodes));
  auto nodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
  CUDA_CHECK(cudaGraphGetNodes(graph, nodes.get(), &numNodes));

  for (size_t i = 0; i < numNodes; i++) {
    std::cout << "Node " << i << ":" << std::endl;

    cudaGraphNode_t u = nodes[i];
    cudaKernelNodeParams params;
    CUDA_CHECK(cudaGraphKernelNodeGetParams(u, &params));

    const char *funcName;
    CUDA_CHECK(cudaFuncGetName(&funcName, params.func));
    std::string s(funcName);
    if (analysisParser.funcNameToKernelInfoMap.count(s) == 0) {
      std::cerr << "Could not find kernel " << s << std::endl;
      abort();
    }

    auto kernelInfo = analysisParser.funcNameToKernelInfoMap[s];

    std::cout << "  Func Name: " << kernelInfo.funcName << std::endl;

    for (const auto &[index, effect] : kernelInfo.ptrArgInfos) {
      std::cout << "  Arg #" << index
                << " : Effect: " << effect
                << " Value: " << (*static_cast<int **>(params.kernelParams[index]))
                << std::endl;
    }
  }
}

__global__ void foo(int a, char b, long long c, int *x, int *y) {
  *x = *y;
  printf("[foo] x -> %p\n", x);
  printf("[foo] y -> %p\n", y);
}

int main() {
  int *x, *y;
  CUDA_CHECK(cudaMallocManaged(&x, sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&y, sizeof(int)));
  printf("x -> %p\n", x);
  printf("y -> %p\n", y);
  *x = 0;
  *y = 1;

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  foo<<<1, 1, 0, s>>>(1, 2, 3, x, y);

  cudaGraph_t g;
  CUDA_CHECK(cudaStreamEndCapture(s, &g));

  analyzeGraph(g);

  return 0;
}