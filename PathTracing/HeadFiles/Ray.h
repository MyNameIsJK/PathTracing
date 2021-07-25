#pragma once
#ifndef RAY_H
#define RAY_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_math.h>

struct Ray
{
	float3 origin;
	float3 direction;
	float3 invDir;
	__device__ __host__ Ray(const float3& o, const float3& d)
	{
		origin = o;
		direction = d;
		invDir = 1.0f / d;
	}
	__device__ __host__ Ray() {}
};
#endif