// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// See LICENSE for license information.
//
// Header-only HIP-graph latency helper that mirrors the Python
// ``magpie.bench.do_bench_cudagraph`` algorithm so HIP testcases produce
// directly comparable wall-clock numbers (same warmup -> estimate ->
// unrolled replay -> median across retries math).
//
// Usage from a .hip / .cpp testcase:
//
//   #define MAGPIE_BENCH_IMPLEMENTATION
//   #include "magpie_bench.hpp"
//
//   int main() {
//     ... setup ...
//     auto stats = magpie::bench::do_bench_hipgraph(
//         [&]() { my_kernel<<<grid, block, 0, magpie::bench::current_stream()>>>(...); },
//         /*rep_ms=*/20, /*n_retries=*/5, /*estimate_reps=*/5);
//     magpie::bench::print_marker(stats);
//     return 0;
//   }
//
// Output (one line, parsed by Magpie's user-harness sub-mode):
//
//   MAGPIE_LATENCY_JSON: {"stats":{"median_ms":...,"p99_ms":...,...}}
//
// Magpie's ``Latency`` stage (Magpie/eval/latency.py) picks this up via
// the same ``MAGPIE_LATENCY_JSON:`` marker contract used for Triton
// kernels; HIP and Triton report identical wall-clock stats.

#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#define MAGPIE_BENCH_HIP_CHECK(call)                                       \
    do {                                                                    \
        hipError_t _err = (call);                                          \
        if (_err != hipSuccess) {                                          \
            throw std::runtime_error(                                      \
                std::string("HIP error: ") + hipGetErrorString(_err));     \
        }                                                                   \
    } while (0)

namespace magpie {
namespace bench {

struct LatencyStats {
    double median_ms = 0.0;
    double p50_ms    = 0.0;
    double p99_ms    = 0.0;
    double min_ms    = 0.0;
    double max_ms    = 0.0;
    double std_ms    = 0.0;
    double estimate_ms = 0.0;
    int    n_repeat  = 0;
    int    n_retries = 0;
    std::vector<double> samples_ms;
};

inline hipStream_t& current_stream() {
    // Side stream used by do_bench_hipgraph; users issuing inside ``fn``
    // should target this stream so the captured graph is well-defined.
    static hipStream_t s = nullptr;
    return s;
}

namespace detail {

inline double percentile(std::vector<double> sorted_v, double p) {
    if (sorted_v.empty()) return 0.0;
    if (sorted_v.size() == 1) return sorted_v[0];
    double rank = p * (sorted_v.size() - 1) / 100.0;
    int    lo   = static_cast<int>(rank);
    int    hi   = std::min(lo + 1, static_cast<int>(sorted_v.size()) - 1);
    double frac = rank - lo;
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * frac;
}

inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline double stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= v.size();
    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    return std::sqrt(var / (v.size() - 1));
}

inline LatencyStats stats_from_samples(const std::vector<double>& samples,
                                       int n_repeat,
                                       int n_retries,
                                       double estimate_ms) {
    LatencyStats s;
    s.n_repeat   = n_repeat;
    s.n_retries  = n_retries;
    s.estimate_ms = estimate_ms;
    s.samples_ms  = samples;
    if (samples.empty()) return s;

    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    s.min_ms    = sorted.front();
    s.max_ms    = sorted.back();
    s.median_ms = median(samples);
    s.p50_ms    = percentile(sorted, 50.0);
    s.p99_ms    = percentile(sorted, 99.0);
    s.std_ms    = stddev(samples);
    return s;
}

} // namespace detail

// Benchmark ``fn`` via HIP-graph estimate-then-unrolled-replay. Mirrors
// ``magpie.bench.do_bench_cudagraph`` byte-for-byte (same n_repeat
// formula, same median across n_retries).
template <typename Fn>
inline LatencyStats do_bench_hipgraph(Fn&& fn,
                                       int rep_ms       = 20,
                                       int n_retries    = 5,
                                       int estimate_reps = 5) {
    hipStream_t& stream = current_stream();
    if (stream == nullptr) {
        MAGPIE_BENCH_HIP_CHECK(hipStreamCreate(&stream));
    }

    // Warmup
    fn();
    MAGPIE_BENCH_HIP_CHECK(hipStreamSynchronize(stream));

    // ----- Step 1: capture estimate graph -----------------------------
    hipGraph_t     est_graph = nullptr;
    hipGraphExec_t est_exec  = nullptr;

    MAGPIE_BENCH_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < estimate_reps; ++i) fn();
    MAGPIE_BENCH_HIP_CHECK(hipStreamEndCapture(stream, &est_graph));
    MAGPIE_BENCH_HIP_CHECK(hipGraphInstantiate(&est_exec, est_graph, nullptr, nullptr, 0));
    MAGPIE_BENCH_HIP_CHECK(hipStreamSynchronize(stream));

    // ----- Step 2: estimate per-call cost -----------------------------
    hipEvent_t e0, e1;
    MAGPIE_BENCH_HIP_CHECK(hipEventCreate(&e0));
    MAGPIE_BENCH_HIP_CHECK(hipEventCreate(&e1));

    MAGPIE_BENCH_HIP_CHECK(hipEventRecord(e0, stream));
    MAGPIE_BENCH_HIP_CHECK(hipGraphLaunch(est_exec, stream));
    MAGPIE_BENCH_HIP_CHECK(hipEventRecord(e1, stream));
    MAGPIE_BENCH_HIP_CHECK(hipStreamSynchronize(stream));
    float est_total_ms = 0.0f;
    MAGPIE_BENCH_HIP_CHECK(hipEventElapsedTime(&est_total_ms, e0, e1));
    double estimate_ms = static_cast<double>(est_total_ms) / estimate_reps;

    int n_repeat;
    if (estimate_ms <= 0.0) {
        n_repeat = 1000;
    } else {
        n_repeat = std::max(1, static_cast<int>(rep_ms / estimate_ms));
    }

    MAGPIE_BENCH_HIP_CHECK(hipGraphExecDestroy(est_exec));
    MAGPIE_BENCH_HIP_CHECK(hipGraphDestroy(est_graph));

    // ----- Step 3: capture timed graph with n_repeat unrolled calls --
    hipGraph_t     timed_graph = nullptr;
    hipGraphExec_t timed_exec  = nullptr;

    MAGPIE_BENCH_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < n_repeat; ++i) fn();
    MAGPIE_BENCH_HIP_CHECK(hipStreamEndCapture(stream, &timed_graph));
    MAGPIE_BENCH_HIP_CHECK(hipGraphInstantiate(&timed_exec, timed_graph, nullptr, nullptr, 0));
    MAGPIE_BENCH_HIP_CHECK(hipStreamSynchronize(stream));

    // ----- Step 4: measure n_retries replays --------------------------
    std::vector<double> samples;
    samples.reserve(n_retries);

    for (int i = 0; i < n_retries; ++i) {
        MAGPIE_BENCH_HIP_CHECK(hipEventRecord(e0, stream));
        MAGPIE_BENCH_HIP_CHECK(hipGraphLaunch(timed_exec, stream));
        MAGPIE_BENCH_HIP_CHECK(hipEventRecord(e1, stream));
        MAGPIE_BENCH_HIP_CHECK(hipStreamSynchronize(stream));
        float total_ms = 0.0f;
        MAGPIE_BENCH_HIP_CHECK(hipEventElapsedTime(&total_ms, e0, e1));
        samples.push_back(static_cast<double>(total_ms) / n_repeat);
    }

    MAGPIE_BENCH_HIP_CHECK(hipEventDestroy(e0));
    MAGPIE_BENCH_HIP_CHECK(hipEventDestroy(e1));
    MAGPIE_BENCH_HIP_CHECK(hipGraphExecDestroy(timed_exec));
    MAGPIE_BENCH_HIP_CHECK(hipGraphDestroy(timed_graph));

    return detail::stats_from_samples(samples, n_repeat, n_retries, estimate_ms);
}

// Print the canonical ``MAGPIE_LATENCY_JSON: {...}`` line to stdout. The
// payload schema matches what ``Magpie/bench/_runner.py`` emits so the
// same parser in ``Magpie/eval/latency.py`` ingests both.
inline void print_marker(const LatencyStats& s,
                         const std::string& kernel_name = "") {
    std::printf(
        "MAGPIE_LATENCY_JSON: "
        "{\"mode\":\"hip_graph\","
        "\"stats\":{\"median_ms\":%.6f,\"p50_ms\":%.6f,\"p99_ms\":%.6f,"
        "\"min_ms\":%.6f,\"max_ms\":%.6f,\"std_ms\":%.6f,"
        "\"n_repeat\":%d,\"n_retries\":%d,\"estimate_ms\":%.6f,"
        "\"samples_ms\":[",
        s.median_ms, s.p50_ms, s.p99_ms,
        s.min_ms, s.max_ms, s.std_ms,
        s.n_repeat, s.n_retries, s.estimate_ms);
    for (size_t i = 0; i < s.samples_ms.size(); ++i) {
        std::printf("%s%.6f", (i ? "," : ""), s.samples_ms[i]);
    }
    std::printf("]}");
    if (!kernel_name.empty()) {
        std::printf(",\"kernel_name\":\"%s\"", kernel_name.c_str());
    }
    std::printf("}\n");
    std::fflush(stdout);
}

} // namespace bench
} // namespace magpie
