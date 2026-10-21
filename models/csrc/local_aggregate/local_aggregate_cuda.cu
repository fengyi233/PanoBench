/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "config.h"
#include "aggregator.h"
#include <fstream>
#include <string>
#include <functional>

// The render kernels are only instantiated for the channel counts listed in
// config.h; anything else would silently leave the output at zero.
static void CHECK_CHANNELS(int C) {
	bool supported = false;
#define LOCALAGG_MATCH_CHANNELS(CHANNELS) supported |= (C == CHANNELS);
	LOCALAGG_SUPPORTED_CHANNELS(LOCALAGG_MATCH_CHANNELS)
#undef LOCALAGG_MATCH_CHANNELS
	TORCH_CHECK(
		supported,
		"local_aggregate was not compiled for ", C,
		" channels. Add the value to LOCALAGG_SUPPORTED_CHANNELS in "
		"models/csrc/local_aggregate/config.h and rebuild.");
}

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateCUDA(
	const torch::Tensor& pts,            // n, 3
	const torch::Tensor& points_int,
	const torch::Tensor& means3D,        // g, 3
	const torch::Tensor& means3D_int,
    const torch::Tensor& opacity,        // g
	const torch::Tensor& semantics,      // g, c
	const torch::Tensor& radii,          // g
	const torch::Tensor& cov3D,          // g, 6
	const int H, int W, int D)
{
  
	const int P = means3D.size(0);
	const int N = pts.size(0);
	const int C = semantics.size(1);
	CHECK_CHANNELS(C);

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_logits = torch::full({N, C}, 0.0, float_opts);
	
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
	
	int rendered;
	rendered = LocalAggregator::Aggregator::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, N, C,
		pts.contiguous().data<float>(),
		points_int.contiguous().data<int>(),
		means3D.contiguous().data<float>(),
		means3D_int.contiguous().data<int>(),
		opacity.contiguous().data<float>(),
		semantics.contiguous().data<float>(),
		cov3D.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		H, W, D,
		out_logits.contiguous().data<float>());
	
	return std::make_tuple(rendered, out_logits, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateBackwardCUDA(
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int H, int W, int D,
	const int R,
	const torch::Tensor& means3D,
	const torch::Tensor& pts,
	const torch::Tensor& points_int,
	const torch::Tensor& cov3D,
	const torch::Tensor& opacities,
	const torch::Tensor& semantics,
	const torch::Tensor& out_grad) 
{
	const int P = means3D.size(0);
	const int N = pts.size(0);
	const int C = semantics.size(1);
	CHECK_CHANNELS(C);

	torch::Tensor means3D_grad = torch::zeros({P, 3}, means3D.options());
	torch::Tensor opacity_grad = torch::zeros({P}, means3D.options());
	torch::Tensor semantics_grad = torch::zeros({P, C}, means3D.options());
	torch::Tensor cov3D_grad = torch::zeros({P, 6}, means3D.options());

	torch::Tensor voxel2pts = torch::full({H * W * D}, -1, means3D.options().dtype(torch::kInt32));
  
	LocalAggregator::Aggregator::backward(
		P, R, N, C,
		H, W, D,
		reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
		reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
        points_int.contiguous().data<int>(),
		voxel2pts.contiguous().data<int>(),
		pts.contiguous().data<float>(),
		means3D.contiguous().data<float>(),
		cov3D.contiguous().data<float>(),
		opacities.contiguous().data<float>(),
		semantics.contiguous().data<float>(),
		out_grad.contiguous().data<float>(),
		means3D_grad.contiguous().data<float>(),
		opacity_grad.contiguous().data<float>(),
		semantics_grad.contiguous().data<float>(),
		cov3D_grad.contiguous().data<float>());

	return std::make_tuple(means3D_grad, opacity_grad, semantics_grad, cov3D_grad);
}
