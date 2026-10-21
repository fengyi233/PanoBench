// Copyright (c) Phigent Robotics. All rights reserved.
// Reference: https://arxiv.org/abs/2211.17111

#include <math.h>

__global__ void bev_pool_v2_kernel(int c, int n_intervals,
                                   const float* __restrict__ depth,
                                   const float* __restrict__ feat,
                                   const int* __restrict__ ranks_depth,
                                   const int* __restrict__ ranks_feat,
                                   const int* __restrict__ ranks_bev,
                                   const int* __restrict__ interval_starts,
                                   const int* __restrict__ interval_lengths,
                                   float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;

  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  float psum = 0;
  for (int i = 0; i < interval_length; i++) {
    const float* cur_depth = depth + ranks_depth[interval_start + i];
    const float* cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    psum += *cur_feat * *cur_depth;
  }

  const int* cur_rank = ranks_bev + interval_start;
  float* cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}

__global__ void bev_pool_v2_grad_kernel(
    int c, int n_intervals, const float* __restrict__ out_grad,
    const float* __restrict__ depth, const float* __restrict__ feat,
    const int* __restrict__ ranks_depth, const int* __restrict__ ranks_feat,
    const int* __restrict__ ranks_bev,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths, float* __restrict__ depth_grad,
    float* __restrict__ feat_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_intervals) return;

  int interval_start = interval_starts[idx];
  int interval_length = interval_lengths[idx];

  for (int i = 0; i < interval_length; i++) {
    const float* cur_out_grad_start = out_grad + ranks_bev[interval_start + i] * c;
    const float* cur_feat_start = feat + ranks_feat[interval_start + i] * c;
    float grad_sum = 0;
    for (int cur_c = 0; cur_c < c; cur_c++) {
      grad_sum += cur_out_grad_start[cur_c] * cur_feat_start[cur_c];
    }
    depth_grad[ranks_depth[interval_start + i]] = grad_sum;
  }

  for (int cur_c = 0; cur_c < c; cur_c++) {
    float grad_sum = 0;
    for (int i = 0; i < interval_length; i++) {
      grad_sum += out_grad[ranks_bev[interval_start + i] * c + cur_c] *
                  depth[ranks_depth[interval_start + i]];
    }
    feat_grad[ranks_feat[interval_start] * c + cur_c] = grad_sum;
  }
}

void bev_pool_v2_cuda_forward(int c, int n_intervals, const float* depth,
                              const float* feat, const int* ranks_depth,
                              const int* ranks_feat, const int* ranks_bev,
                              const int* interval_starts,
                              const int* interval_lengths, float* out) {
  bev_pool_v2_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
      c, n_intervals, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, out);
}

void bev_pool_v2_cuda_backward(int c, int n_intervals, const float* out_grad,
                               const float* depth, const float* feat,
                               const int* ranks_depth, const int* ranks_feat,
                               const int* ranks_bev,
                               const int* interval_starts,
                               const int* interval_lengths, float* depth_grad,
                               float* feat_grad) {
  bev_pool_v2_grad_kernel<<<(int)ceil(((double)n_intervals / 256)), 256>>>(
      c, n_intervals, out_grad, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, depth_grad, feat_grad);
}
