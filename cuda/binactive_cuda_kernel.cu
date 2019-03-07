#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

namespace {
template <typename scalar_t>
__global__ void binactive_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // output[idx] = fminf(fmaxf(delta*floor((input[idx] / delta) + 0.5), minv), maxv);
    output[idx] = (input[idx]>0)?1:(input[idx]<0?-1:0);
  }
}

template <typename scalar_t>
__global__ void binactive_cuda_backward_kernel(
    scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ input,
    const int32_t size) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const float inp = input[idx];
  if (idx < size) {
      grad_h[idx] *= (inp > -1)*(inp < 1);
  }
}
} // namespace

at::Tensor binactive_cuda_forward(
    const at::Tensor input) {
  const auto size = input.size(0)*input.size(1)*input.size(2)*input.size(3);
  auto output = at::zeros_like(input);

  const int32_t threads = 1024;
  const int32_t blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "binactive_forward_cuda", ([&] {
    binactive_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        output.data<scalar_t>(),
        size);
  }));

  return output;
}

at::Tensor binactive_cuda_backward(
    at::Tensor grad_h,
    const at::Tensor input) {

  const auto size = input.size(0)*input.size(1)*input.size(2)*input.size(3);
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "binactive_backward_cuda", ([&] {
    binactive_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_h.data<scalar_t>(),
        input.data<scalar_t>(),
        size);
  }));

  return grad_h;
}
