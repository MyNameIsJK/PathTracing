#pragma once
#ifndef INTERSECT_H
#define INTERSECT_H

#include "BoundingBox.h"
#include "Triangle.h"
#include "BVH.h"
#include "Ray.h"
#include "GlobalFunc.h"

inline __device__ float determinant(const float3& A, const float3& B, const float3& C)
{
	return dot(cross(C, A), B);
}
inline __device__ bool rayTriangleIntersect(const Ray& ray, const Triangle& triangle, float& tnear)
{
	float3 D = ray.origin - triangle.v2;
	float3 A = triangle.v0 - triangle.v2;
	float3 B = triangle.v1 - triangle.v2;
	float3 C = -1.0f * ray.direction;

	float down = determinant(A, B, C);
	if (fabs(down) <= 0.005f)
		return false;
	float a = determinant(D, B, C) / down;
	float b = determinant(A, D, C) / down;
	float t = determinant(A, B, D) / down;
	if (t < 0.005f)
		return false;
	if (a < 0.0f || b < 0.0f)
		return false;
	if (a + b > 1.0f)
		return false;
	tnear = t;
	return true;
}
/*
inline __device__ bool raySphereIntersect(const Ray& ray, const Sphere& sphere, float& tnear)
{
	float3 eyeToCenter = ray.origin - sphere.pos;
	float a = dot(ray.dir, ray.dir);
	float b = 2.0f * dot(eyeToCenter, ray.dir);
	float c = dot(eyeToCenter, eyeToCenter) - sphere.radius * sphere.radius;
	tnear = 0.0f;
	float delta = b * b - 4 * a * c;
	if (delta < 0.0f)
		return false;
	delta = sqrtf(delta);
	float t1 = -b + delta;
	t1 /= 2.0f;
	t1 /= a;
	float t2 = -b - delta;
	t2 /= 2.0f;
	t2 /= a;
	if (delta == 0.0f)
	{
		if (t2 < 0.0f)
			return false;
		else
			tnear = t2;
	}
	else
	{
		if (t2 > 0.0f)
			tnear = t2;
		else if (t1 > 0.0f)
			tnear = t1;
		else
			return false;
	}
	return true;
}
*/
inline __device__ bool rayBoundingBoxIntersect(const Ray& ray, const BoundingBox& AABB, float& tnear)
{
	float3 invR = make_float3(1.0f) / ray.direction;
	float3 tbot = invR * (AABB.minCorner - ray.origin);
	float3 ttop = invR * (AABB.maxCorner - ray.origin);
	float3 tmin = fminf(tbot, ttop);
	float3 tmax = fmaxf(ttop, tbot);
	float largestTmin = maxf(tmin);
	float smallestTmax = minf(tmax);
	tnear = largestTmin;
	return smallestTmax+0.01f >= largestTmin;
}
/*
inline __device__ bool rayBVHIntersect(const Ray& ray, BVHArrNode* bvhArr, Triangle*triangles, float &tnear, int&hitIndex)
{
	int curIndex = 0;
	if (rayBoundingBoxIntersect(ray, bvhArr[0].AABB, tnear))
	{
		int layer = 1;
		while (curIndex >= 0)
		{
			layer++;
			if (bvhArr[curIndex].triangleIndex != -1)
			{
				if (rayTriangleIntersect(ray, triangles[bvhArr[curIndex].triangleIndex], tnear))
				{
					hitIndex = bvhArr[curIndex].triangleIndex;
					return true;
				}
				else
				{
					hitIndex = 28;
					return true;
				}
			}
			else
			{
				float tleft, tright;
				bool ileft, iright;
				ileft = rayBoundingBoxIntersect(ray, bvhArr[curIndex + 1].AABB, tleft);
				iright = rayBoundingBoxIntersect(ray, bvhArr[bvhArr[curIndex].rightChildIndex].AABB, tright);
				if (ileft == false && iright == true)
					curIndex = bvhArr[curIndex].rightChildIndex;
				else if (ileft == true && iright == false)
					curIndex++;
				else if (ileft == true && iright == true)
				{
					if (tleft < tright)
						curIndex++;
					else
						curIndex = bvhArr[curIndex].rightChildIndex;
				}
				else
				{
					if (layer == 3)
					{
						hitIndex = 26;
						return true;
					}
					return false;
				}
			}
		}
	}
	else
		return false;	
}
*/
#endif