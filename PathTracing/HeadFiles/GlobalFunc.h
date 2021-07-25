#pragma once
#ifndef GLOBAL_FUNC_H
#define GLOBAL_FUNC_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_math.h>

inline __device__ __host__ float maxf(const float3& f3)
{
	return max(max(f3.x, f3.y), f3.z);
}
inline __device__ __host__ float minf(const float3& f3)
{
	return min(min(f3.x, f3.y), f3.z);
}
inline __device__ __host__ bool operator < (const float3& a, const float3& b)
{
	if (a.x < b.x && a.y < b.y && a.z < b.z)
		return true;
	return false;
}
inline __device__ __host__ bool operator <= (const float3& a, const float3& b)
{
	if (a.x <= b.x && a.y <= b.y && a.z <= b.z)
		return true;
	return false;
}
inline __device__ __host__ bool operator > (const float3& b, const float3& a)
{
	if (a.x < b.x && a.y < b.y && a.z < b.z)
		return true;
	return false;
}
inline __device__ __host__ bool operator >= (const float3& b, const float3& a)
{
	if (a.x <= b.x && a.y <= b.y && a.z <= b.z)
		return true;
	return false;
}
#endif