#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/util/complex.h>


template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, int pack_size>
union Pack {
  __device__ Pack() {
    // do nothing
  }
  PackType<T, pack_size> storage;
  T elem[pack_size];
};

 
__global__ void int82half_kernel(Pack<int8_t, 8>* input, Pack<c10::Half, 8>* output, float idx_value, float scale) {
  long idx = threadIdx.x + blockIdx.x * blockDim.x;
  Pack<int8_t, 8> data = input[idx];
  Pack<c10::Half, 8> result ;
    
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    c10::Half half_data = static_cast<c10::Half>(data.elem[i]);
    c10::Half sign = (half_data >= 0) ? 1 : -1;
    c10::Half output = sign * pow((1/scale) * abs(half_data), idx_value);
    result.elem[i] = output;
  }
 
  __syncthreads();
  output[idx] = result;
}


void int82half(torch::Tensor& tensor_in, torch::Tensor& tensor_out , float idx_value, float scale) {
   
  //开启线程数量
  long size = tensor_in.size(0) / 8;
  // 获取输入 Tensor 的指针
  Pack<int8_t, 8>* input_data = reinterpret_cast<Pack<int8_t, 8>*>(tensor_in.data_ptr());
  // 获取输出 Tensor 的指针
  Pack<c10::Half, 8>* output_data = reinterpret_cast<Pack<c10::Half, 8>*>(tensor_out.data_ptr());
  // 计算 grid 大小
  int blockDIM = 256;
  int gridDIM = size / blockDIM;
  // 调用 CUDA kernel
  int82half_kernel<<<gridDIM, blockDIM>>>(input_data, output_data, idx_value , scale);
  // 等待 GPU 完成操作
  cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int82half", &int82half, "CUDA kernel: int82half");
}
