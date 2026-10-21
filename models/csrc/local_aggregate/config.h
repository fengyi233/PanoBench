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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// The render kernels keep the per-point accumulator in registers, so the
// channel count has to be a compile-time constant. Instead of hardcoding one
// value we instantiate the kernel for every class count PanoBench ships and
// pick the instantiation at launch time. Add the value here when introducing a
// dataset with a new number of classes.
#define LOCALAGG_SUPPORTED_CHANNELS(MACRO) \
	MACRO(17)                              \
	MACRO(18)                              \
	MACRO(19)                              \
	MACRO(20)                              \
	MACRO(30)

#endif
