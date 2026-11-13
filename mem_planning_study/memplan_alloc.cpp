// Planned CUDA allocator for the memory-planning study (CUDAPluggableAllocator).
//
// Build (run.py loads memplan_alloc.so from this directory):
//   g++ -O2 -std=c++17 -shared -fPIC memplan_alloc.cpp -I/usr/local/cuda/include \
//       -L/usr/local/cuda/lib64 -lcudart -o memplan_alloc.so
//
// Every allocation is either
//   - planned: placed at a precomputed offset inside one preallocated arena,
//     matched by its ordinal within the current "window" (one train step); or
//   - fallback: served by the CUDA stream-ordered allocator (cudaMallocAsync)
//     from the device default pool with an unlimited release threshold.
//
// A planned placement is only used when the requested (512B-rounded) size
// matches the plan, the request comes from the window's thread and stream,
// and the target range does not overlap a live arena block. Any size
// mismatch marks the window as diverged and the rest of it falls back.
//
// Trace mode records the alloc/free sequence of one chosen window; the
// offline planner turns it into a plan file.

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr int64_t kAlign = 512;
inline int64_t round_up(int64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

struct PlanEntry {
  int64_t size;
  int64_t offset;  // -1 => fallback
};

struct TraceEvent {
  int32_t window;
  int32_t kind;  // 1 alloc, 0 free
  int32_t on_window_thread;
  int64_t ordinal;  // alloc ordinal within the window, -1 otherwise
  int64_t size;
  uint64_t ptr;
  uint64_t stream;
};

std::mutex mu;
bool plan_active = false;

// Window state.
bool in_window = false;
int32_t window_id = -1;
int64_t ordinal = 0;
std::thread::id window_tid;
uint64_t window_stream = 0;
bool diverged = false;

// Trace: windows [trace_window, trace_window + trace_count).
int32_t trace_window = -2;
int32_t trace_count = 0;
inline bool tracing() {
  return in_window && window_id >= trace_window &&
      window_id < trace_window + trace_count;
}
std::vector<TraceEvent> trace;

// Plan.
std::vector<PlanEntry> plan;
char* arena = nullptr;
int64_t arena_size = 0;
std::map<uintptr_t, int64_t> live_arena;  // start -> size

// Stats.
int64_t cur_alloc = 0, peak_alloc = 0;
int64_t arena_live = 0, arena_live_peak = 0;
int64_t hits = 0, misses = 0, overlaps = 0, window_fallbacks = 0;
int64_t last_window_allocs = 0;

cudaMemPool_t pool = nullptr;

void ensure_pool(int device) {
  if (pool != nullptr) {
    return;
  }
  cudaDeviceGetDefaultMemPool(&pool, device);
  uint64_t threshold = UINT64_MAX;
  cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

bool overlaps_live(uintptr_t start, int64_t size) {
  auto it = live_arena.lower_bound(start);
  if (it != live_arena.end() && it->first < start + size) {
    return true;
  }
  if (it != live_arena.begin()) {
    --it;
    if (it->first + it->second > start) {
      return true;
    }
  }
  return false;
}

}  // namespace

extern "C" {

void* memplan_malloc(ssize_t size, int device, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mu);
  if (size == 0) {
    return nullptr;
  }
  ensure_pool(device);
  const int64_t rsize = round_up(size);
  void* p = nullptr;
  int64_t ord = -1;
  const bool window_thread = in_window && std::this_thread::get_id() == window_tid;
  if (window_thread) {
    ord = ordinal++;
    if (plan_active && !diverged) {
      if (ord >= static_cast<int64_t>(plan.size()) || plan[ord].size != rsize) {
        diverged = true;
        ++misses;
      } else if (plan[ord].offset >= 0 &&
                 reinterpret_cast<uint64_t>(stream) == window_stream) {
        uintptr_t start = reinterpret_cast<uintptr_t>(arena) + plan[ord].offset;
        if (overlaps_live(start, rsize)) {
          ++overlaps;
        } else {
          p = reinterpret_cast<void*>(start);
          live_arena.emplace(start, rsize);
          arena_live += rsize;
          if (arena_live > arena_live_peak) {
            arena_live_peak = arena_live;
          }
          ++hits;
        }
      }
    }
  }
  if (p == nullptr) {
    if (cudaMallocAsync(&p, rsize, stream) != cudaSuccess) {
      cudaGetLastError();
      return nullptr;
    }
    if (window_thread) {
      ++window_fallbacks;
    }
  }
  cur_alloc += rsize;
  if (cur_alloc > peak_alloc) {
    peak_alloc = cur_alloc;
  }
  if (tracing()) {
    trace.push_back({window_id, 1, window_thread ? 1 : 0, ord, rsize,
                     reinterpret_cast<uint64_t>(p),
                     reinterpret_cast<uint64_t>(stream)});
  }
  return p;
}

void memplan_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mu);
  if (ptr == nullptr) {
    return;
  }
  const int64_t rsize = round_up(size);
  cur_alloc -= rsize;
  uintptr_t u = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t a = reinterpret_cast<uintptr_t>(arena);
  if (arena != nullptr && u >= a && u < a + arena_size) {
    live_arena.erase(u);
    arena_live -= rsize;
  } else {
    cudaFreeAsync(ptr, stream);
  }
  if (tracing()) {
    trace.push_back({window_id, 0, std::this_thread::get_id() == window_tid ? 1 : 0, -1,
                     rsize, reinterpret_cast<uint64_t>(ptr),
                     reinterpret_cast<uint64_t>(stream)});
  }
}

void memplan_begin_window(uint64_t stream) {
  std::lock_guard<std::mutex> lock(mu);
  in_window = true;
  ++window_id;
  ordinal = 0;
  diverged = false;
  window_tid = std::this_thread::get_id();
  window_stream = stream;
  hits = misses = overlaps = window_fallbacks = 0;
}

void memplan_end_window() {
  std::lock_guard<std::mutex> lock(mu);
  in_window = false;
  last_window_allocs = ordinal;
}

void memplan_set_trace_window(int32_t w, int32_t count) {
  std::lock_guard<std::mutex> lock(mu);
  trace_window = w;
  trace_count = count;
}

int memplan_dump_trace(const char* path) {
  std::lock_guard<std::mutex> lock(mu);
  std::ofstream f(path);
  if (!f) {
    return -1;
  }
  f << "# window kind on_window_thread ordinal size ptr stream\n";
  for (const auto& e : trace) {
    f << e.window << ' ' << e.kind << ' ' << e.on_window_thread << ' ' << e.ordinal << ' ' << e.size
      << ' ' << e.ptr << ' ' << e.stream << '\n';
  }
  return static_cast<int>(trace.size());
}

// Plan file: first line "arena_size num_entries", then one "size offset" line
// per window alloc ordinal.
int memplan_load_plan(const char* path, int device) {
  std::lock_guard<std::mutex> lock(mu);
  std::ifstream f(path);
  if (!f) {
    return -1;
  }
  int64_t n = 0;
  f >> arena_size >> n;
  plan.resize(n);
  for (int64_t i = 0; i < n; ++i) {
    f >> plan[i].size >> plan[i].offset;
  }
  cudaSetDevice(device);
  if (cudaMalloc(reinterpret_cast<void**>(&arena), arena_size) != cudaSuccess) {
    cudaGetLastError();
    arena = nullptr;
    return -2;
  }
  return static_cast<int>(n);
}

void memplan_set_plan_active(int on) {
  std::lock_guard<std::mutex> lock(mu);
  plan_active = on != 0 && arena != nullptr;
}

void memplan_trim_pool() {
  cudaDeviceSynchronize();
  if (pool != nullptr) {
    cudaMemPoolTrimTo(pool, 0);
  }
}

void memplan_reset_peak() {
  std::lock_guard<std::mutex> lock(mu);
  peak_alloc = cur_alloc;
  arena_live_peak = arena_live;
  if (pool != nullptr) {
    uint64_t zero = 0;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &zero);
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero);
  }
}

// out: [cur_alloc, peak_alloc, arena_size, arena_live_peak,
//       pool_reserved_cur, pool_reserved_high, pool_used_cur, pool_used_high,
//       hits, misses, overlaps, window_fallbacks, last_window_allocs, diverged]
void memplan_stats(int64_t* out) {
  std::lock_guard<std::mutex> lock(mu);
  uint64_t rc = 0, rh = 0, uc = 0, uh = 0;
  if (pool != nullptr) {
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &rc);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &rh);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &uc);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &uh);
  }
  int64_t vals[] = {cur_alloc, peak_alloc, arena_size, arena_live_peak,
                    static_cast<int64_t>(rc), static_cast<int64_t>(rh),
                    static_cast<int64_t>(uc), static_cast<int64_t>(uh),
                    hits, misses, overlaps, window_fallbacks,
                    last_window_allocs, diverged ? 1 : 0};
  for (int i = 0; i < 14; ++i) {
    out[i] = vals[i];
  }
}

}  // extern "C"
