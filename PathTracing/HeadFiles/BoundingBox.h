#pragma once
#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_math.h>

enum Axis
{
	X, Y, Z
};
class BoundingBox
{
public:
	float3 maxCorner;
	float3 minCorner;
public:
	BoundingBox();
	BoundingBox(const float3& maxCor, const float3& minCor);
	BoundingBox(const BoundingBox& AABB);

	Axis calculateLongestAxis();
	bool isPointInside(const float3& p)const;
	float area();//sah
	float3 getOffset(const float3& p)const;
	float3 getCnetroid()const;

	__host__ __device__ BoundingBox& operator=(const BoundingBox& AABB)
	{
		maxCorner = AABB.maxCorner;
		minCorner = AABB.minCorner;
		return *this;
	}
	__host__ __device__ const float3& operator[](int i)const
	{
		if (i == 0)
			return minCorner;
		else
			return maxCorner;
	}
	__host__ __device__ BoundingBox& operator || (const BoundingBox& AABB)
	{
		return BoundingBox(fmaxf(maxCorner, AABB.maxCorner), fminf(minCorner, AABB.minCorner));
	}
	__host__ __device__ BoundingBox& operator || (const float3& p)
	{
		return BoundingBox(fmaxf(maxCorner, p), fminf(minCorner, p));
	}
};

#endif