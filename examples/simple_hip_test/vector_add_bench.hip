// Simple HIP vector_add wrapped in the Magpie HIP-graph latency helper
// for strict parity with the Triton wall-clock numbers.
//
// Compile:
//   hipcc -g -O2 -std=c++17 \
//     -I/path/to/Magpie/Magpie/bench/include \
//     -o vector_add_bench vector_add_bench.hip
//
// Run via Magpie's user-harness sub-mode (latency.bench_target unset,
// testcase_command points at this binary). The binary prints exactly:
//   MAGPIE_LATENCY_JSON: {"mode":"hip_graph","stats":{...}}
// which Magpie/eval/latency.py parses directly.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "magpie_bench.hpp"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20;  // 1M elements — same as the Triton example
    const size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    // Use the Magpie helper's side stream so the captured graph is well-defined.
    hipStream_t& stream = magpie::bench::current_stream();
    if (stream == nullptr) hipStreamCreate(&stream);

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    auto stats = magpie::bench::do_bench_hipgraph(
        [&]() {
            hipLaunchKernelGGL(vector_add, dim3(gridSize), dim3(blockSize),
                               0, stream, d_a, d_b, d_c, N);
        },
        /*rep_ms=*/20,
        /*n_retries=*/5,
        /*estimate_reps=*/5);

    magpie::bench::print_marker(stats, "vector_add");

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    return 0;
}
