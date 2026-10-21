// Copyright (c) Phigent Robotics. All rights reserved.
// Reference: https://arxiv.org/abs/2211.17111

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

void bev_pool_v2_cuda_forward(int c, int n_intervals, const float* depth,
                              const float* feat, const int* ranks_depth,
                              const int* ranks_feat, const int* ranks_bev,
                              const int* interval_starts,
                              const int* interval_lengths, float* out);

void bev_pool_v2_cuda_backward(int c, int n_intervals, const float* out_grad,
                               const float* depth, const float* feat,
                               const int* ranks_depth, const int* ranks_feat,
                               const int* ranks_bev,
                               const int* interval_starts,
                               const int* interval_lengths, float* depth_grad,
                               float* feat_grad);

void bev_pool_v2_forward(const at::Tensor depth, const at::Tensor feat,
                         at::Tensor out, const at::Tensor ranks_depth,
                         const at::Tensor ranks_feat,
                         const at::Tensor ranks_bev,
                         const at::Tensor interval_lengths,
                         const at::Tensor interval_starts) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(depth));
  int c = feat.size(4);
  int n_intervals = interval_lengths.size(0);
  bev_pool_v2_cuda_forward(
      c, n_intervals, depth.data_ptr<float>(), feat.data_ptr<float>(),
      ranks_depth.data_ptr<int>(), ranks_feat.data_ptr<int>(),
      ranks_bev.data_ptr<int>(), interval_starts.data_ptr<int>(),
      interval_lengths.data_ptr<int>(), out.data_ptr<float>());
}

void bev_pool_v2_backward(const at::Tensor out_grad, at::Tensor depth_grad,
                          at::Tensor feat_grad, const at::Tensor depth,
                          const at::Tensor feat,
                          const at::Tensor ranks_depth,
                          const at::Tensor ranks_feat,
                          const at::Tensor ranks_bev,
                          const at::Tensor interval_lengths,
                          const at::Tensor interval_starts) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out_grad));
  int c = out_grad.size(4);
  int n_intervals = interval_lengths.size(0);
  bev_pool_v2_cuda_backward(
      c, n_intervals, out_grad.data_ptr<float>(), depth.data_ptr<float>(),
      feat.data_ptr<float>(), ranks_depth.data_ptr<int>(),
      ranks_feat.data_ptr<int>(), ranks_bev.data_ptr<int>(),
      interval_starts.data_ptr<int>(), interval_lengths.data_ptr<int>(),
      depth_grad.data_ptr<float>(), feat_grad.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_pool_v2_forward", &bev_pool_v2_forward, "bev_pool_v2_forward");
  m.def("bev_pool_v2_backward", &bev_pool_v2_backward, "bev_pool_v2_backward");
}
