#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor binactive_cuda_forward(
    const at::Tensor input);

at::Tensor binactive_cuda_backward(
    at::Tensor grad_h,
    const at::Tensor input);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor binactive_forward(
    const at::Tensor input) {
  CHECK_INPUT(input);
  return binactive_cuda_forward(input);
}

at::Tensor binactive_backward(
    at::Tensor grad_h,
    const at::Tensor input) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input);
  return binactive_cuda_backward(grad_h, input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &binactive_forward, "binactive forward (CUDA)");
  m.def("backward", &binactive_backward, "binactive backward (CUDA)");
}
