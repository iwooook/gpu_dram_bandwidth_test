#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>

// using DataType = float;

// #if 0
// using float16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
// using VecType = float16;
// #else
// using VecType = float4;
// #endif

using DataType = int;

#if 0
using int16 = __attribute__((__vector_size__(16 * sizeof(int)))) int;
using VecType = int16;
#else
using VecType = int4;
#endif

#define CHECK_HIP_ERROR(call)                                                  \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define N_THREADS_PER_BLOCK 256

template <typename T>
__global__ void ReadWriteKernel(T *a, T *b, size_t N, size_t num_chunks) {
  extern __shared__ T _tmp[];
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t tid = threadIdx.x;
  size_t chunk_size = N / num_chunks; // stride

  // old version
  /*
  if (idx < chunk_size) {
      // Read
      for (size_t i = 0; i < num_chunks; i++) {
          _tmp[i * N_THREADS_PER_BLOCK + tid].x = a[i * chunk_size + idx].x;
          _tmp[i * N_THREADS_PER_BLOCK + tid].y = a[i * chunk_size + idx].y;
          _tmp[i * N_THREADS_PER_BLOCK + tid].z = a[i * chunk_size + idx].z;
          _tmp[i * N_THREADS_PER_BLOCK + tid].w = a[i * chunk_size + idx].w;
      }
      __syncthreads();
      //Write
      for (size_t i = 0; i < num_chunks; i++) {
          b[i * chunk_size + idx].x = _tmp[i * N_THREADS_PER_BLOCK + tid].x;
          b[i * chunk_size + idx].y = _tmp[i * N_THREADS_PER_BLOCK + tid].y;
          b[i * chunk_size + idx].z = _tmp[i * N_THREADS_PER_BLOCK + tid].z;
          b[i * chunk_size + idx].w = _tmp[i * N_THREADS_PER_BLOCK + tid].w;
      }
  }
  */

  constexpr size_t VEC_WIDTH = sizeof(T) / sizeof(DataType);
  size_t vec_idx = idx / VEC_WIDTH;

  if (idx < chunk_size) {
    // Read
    for (size_t i = 0; i < num_chunks; i++) {
      T *src = reinterpret_cast<T *>(a + i * chunk_size + idx);
      T *dst = reinterpret_cast<T *>(_tmp + i * N_THREADS_PER_BLOCK + tid);
      dst[0] = src[0];
    }
    __syncthreads();
    // Write
    for (size_t i = 0; i < num_chunks; i++) {
      T *src = reinterpret_cast<T *>(_tmp + i * N_THREADS_PER_BLOCK + tid);
      T *dst = reinterpret_cast<T *>(b + i * chunk_size + idx);
      dst[0] = src[0];
    }
  }
}

template <typename T>
__global__ void ReadKernel(T *a, T *b, size_t N, size_t num_chunks) {
  extern __shared__ T _tmp[];
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t tid = threadIdx.x;
  size_t chunk_size = N / num_chunks; // stride

  constexpr size_t VEC_WIDTH = sizeof(T) / sizeof(DataType);
  size_t vec_idx = idx / VEC_WIDTH;

  if (idx < chunk_size) {
    for (size_t i = 0; i < num_chunks; i++) {
      T *src = reinterpret_cast<T *>(a + i * chunk_size + idx);
      T *dst = reinterpret_cast<T *>(_tmp + i * N_THREADS_PER_BLOCK + tid);
      dst[0] = src[0];
    }
  }
}

template <typename T>
__global__ void WriteKernel(T *a, T *b, size_t N, size_t num_chunks) {
  extern __shared__ T _tmp[];
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t tid = threadIdx.x;
  size_t chunk_size = N / num_chunks; // stride

  constexpr size_t VEC_WIDTH = sizeof(T) / sizeof(DataType);
  size_t vec_idx = idx / VEC_WIDTH;

  if (idx < chunk_size) {
    for (size_t i = 0; i < num_chunks; i++) {
      T *src = reinterpret_cast<T *>(_tmp + i * N_THREADS_PER_BLOCK + tid);
      T *dst = reinterpret_cast<T *>(b + i * chunk_size + idx);
      dst[0] = src[0];
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "Usage: ./mem <num_elems> <num_blocks> <num_devs> "
                 "<num_chunks> <num_iter> <test_type>"
              << std::endl;
    return 1;
  }

  // Parse input arguments
  size_t num_elems = std::atoi(argv[1]) * 1024 * 1024;
  size_t num_blocks = std::atoi(argv[2]);
  size_t num_devs = std::atoi(argv[3]);
  size_t num_chunks = std::atoi(argv[4]);
  size_t num_iter = std::atoi(argv[5]);
  size_t test_type = std::atoi(argv[6]);

  std::cout << "num_elems: " << num_elems << ", num_blocks=" << num_blocks
            << ", num_devs=" << num_devs << ", num_chunks=" << num_chunks
            << ", num_iter=" << num_iter << std::endl;

  // kernel function pointer
  void (*kernel)(VecType *, VecType *, size_t, size_t);

  if (test_type == 0) {
    std::cout << "testing: read, write" << std::endl;
    kernel = ReadWriteKernel<VecType>;
  } else if (test_type == 1) {
    std::cout << "testing: read only" << std::endl;
    kernel = ReadKernel<VecType>;
  } else if (test_type == 2) {
    std::cout << "testing: write only" << std::endl;
    kernel = WriteKernel<VecType>;
  } else {
    std::cerr << "Invalid test type" << std::endl;
    return 1;
  }

  std::cout << "size per block = "
            << (float)num_elems * sizeof(VecType) / 1024 / 1024 << " MB"
            << std::endl;

  // Allocate host and device memory
  VecType *h_data = new VecType[num_elems];
  VecType *d_data_a;
  VecType *d_data_b;
  CHECK_HIP_ERROR(hipMalloc(&d_data_a, num_elems * sizeof(VecType)));
  CHECK_HIP_ERROR(hipMalloc(&d_data_b, num_elems * sizeof(VecType)));

  // Initialize host data
  for (size_t i = 0; i < num_elems; i++) {
    //
  }

  // Copy data from host to device
  CHECK_HIP_ERROR(hipMemcpy(d_data_a, h_data, num_elems * sizeof(VecType),
                            hipMemcpyHostToDevice));

  const int threads_per_block = N_THREADS_PER_BLOCK;
  dim3 grids((num_elems + threads_per_block - 1) / threads_per_block /
             num_chunks);
  dim3 blocks(threads_per_block);

  // Warm-up kernel launch
  std::cout << "Warming up..." << std::endl;
  for (size_t i = 0; i < 5; i++) {
    hipLaunchKernelGGL(kernel, grids, blocks,
                       threads_per_block * num_chunks * sizeof(VecType), 0,
                       d_data_a, d_data_b, num_elems, num_chunks);
  }
  CHECK_HIP_ERROR(hipDeviceSynchronize());
  std::cout << "Warming up done." << std::endl;

  // Benchmark kernel launch
  std::cout << "Evaluating..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_iter; i++) {
    hipLaunchKernelGGL(kernel, grids, blocks,
                       threads_per_block * num_chunks * sizeof(VecType), 0,
                       d_data_a, d_data_b, num_elems, num_chunks);
  }
  CHECK_HIP_ERROR(hipDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Evaluation done." << std::endl;

  // Calculate elapsed time
  std::chrono::duration<double> elapsed = end - start;

  // Calculate bandwidth (in GB/s)
  double total_bytes =
      static_cast<double>(num_elems * sizeof(VecType) * num_iter);
  if (test_type == 0) {
    total_bytes *= 2;
  }
  double bandwidth = total_bytes / elapsed.count() / 1e9;

  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
  std::cout << "Global memory bandwidth: " << bandwidth << " GB/s" << std::endl;

  // Cleanup
  CHECK_HIP_ERROR(hipFree(d_data_a));
  CHECK_HIP_ERROR(hipFree(d_data_b));
  delete[] h_data;

  return 0;
}
