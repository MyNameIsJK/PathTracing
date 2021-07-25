#pragma once
#include "Material.h"
#include <vector>
#include "BoundingBox.h"
#include <string>
#include "GlobalDefine.h"
class Triangle
{
public:
	float3 v0, v1, v2;//vertex A,B,C
	float3 e1, e2;//edge v1-v0,v2-v0
	float3 t0, t1, t2;//texture coord
	float3 normal;
	float area;
	Material material;
	Triangle() = default;
	Triangle(float3 _v0, float3 _v1, float3 _v2, Material* m = nullptr);
	__device__ __host__ BoundingBox getBoundingBox()
	{
		float3 minCorner = make_float3(FLT_MAX);
		float3 maxCorner = make_float3(FLT_MIN);
		minCorner = fminf(minCorner, v0);
		minCorner = fminf(minCorner, v1);
		minCorner = fminf(minCorner, v2);
		maxCorner = fmaxf(maxCorner, v0);
		maxCorner = fmaxf(maxCorner, v1);
		maxCorner = fmaxf(maxCorner, v2);
		return BoundingBox(maxCorner, minCorner);
	}
	__device__ __host__ bool hasEmission()
	{
		return length(material.emission) > EPSILON;
	}
	float getArea();
};
class MeshTriangle
{
public:
	std::vector<Triangle>triangles;
	Material* material;
public:
	MeshTriangle(const std::string& fileName, Material* m = new Material);
};

