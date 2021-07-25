#pragma once
#include "Triangle.h"
#include <curand.h>
#include <curand_kernel.h>
__device__ void sampleTriangle(float3& coord, float3& normal, float& pdf, Triangle* triangles, uint index, curandState& randState)
{
	float x = sqrtf(curand_uniform(&randState)), y = curand_uniform(&randState);
	coord = triangles[index].v0 * (1.0f - x) + triangles[index].v1 * (x * (1.0f - y)) + triangles[index].v2 * (x * y);
	normal = triangles[index].normal;
	pdf = 1.0f / triangles[index].area;
}
__device__ bool sampleLight(float3& coord, float3& normal, float& pdf,
	uint* sources, uint numSources, curandState& randState, float* preSumArea,
	Triangle* triangles)
{
	if (numSources == 0)
		return false;
	float aimArea = curand_uniform(&randState) * preSumArea[numSources - 1];
	for (int i = 0; i < numSources; i++)
	{
		if (preSumArea[i] >= aimArea)
		{
			sampleTriangle(coord, normal, pdf, triangles, sources[i], randState);
			return true;
		}
	}
}
__device__ float3 sampleDiffuseMaterial(const float3& normal, curandState& randState)
{
	float x = curand_uniform(&randState);
	float y = curand_uniform(&randState);
	float z = fabsf(1.0f - 2.0f * x);
	float r = sqrtf(1.0f - z * z);
	float phi = z * M_PI * y;
	// 确保localRay为单位向量且z大于0
	float3 localRay = make_float3(r * cos(phi), r * sin(phi), z);

	float3 B, C;
	if (fabsf(normal.x) > fabsf(normal.y))
	{
		float invLen = 1.0f / sqrtf(normal.x * normal.x + normal.z * normal.z);
		C = make_float3(normal.z * invLen, 0.0f, -normal.x * invLen);
	}
	else
	{
		float invLen = 1.0f / sqrtf(normal.y * normal.y + normal.z * normal.z);
		C = make_float3(0.0f, normal.z * invLen, -normal.y * invLen);
	}
	B = cross(C, normal);
	return (localRay.x * B + localRay.y * C + localRay.z * normal);
}