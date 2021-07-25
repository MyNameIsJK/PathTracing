#include "BoundingBox.h"
#include "GlobalFunc.h"
BoundingBox::BoundingBox()
{
	maxCorner = make_float3(0.0f);
	minCorner = maxCorner;
}

BoundingBox::BoundingBox(const float3& maxCor, const float3& minCor)
{
	maxCorner = maxCor;
	minCorner = minCor;
}

BoundingBox::BoundingBox(const BoundingBox& AABB)
{
	maxCorner = AABB.maxCorner;
	minCorner = AABB.minCorner;
}

Axis BoundingBox::calculateLongestAxis()
{
	float3 lenAABB = maxCorner - minCorner;
	if (lenAABB.x > lenAABB.y)
	{
		if (lenAABB.x > lenAABB.z)
			return X;
		else
			return Z;
	}
	else
	{
		if (lenAABB.y > lenAABB.z)
			return Y;
		else
			return Z;
	}
}

bool BoundingBox::isPointInside(const float3& p) const
{
	if (p <= maxCorner && p >= minCorner)
		return true;
	return false;
}

float BoundingBox::area()
{
	float3 dist = maxCorner - minCorner;
	return 2.0f * (dist.x * dist.y + dist.y * dist.z + dist.z * dist.x);
}

float3 BoundingBox::getOffset(const float3& p) const
{
	float3 offset = p - minCorner;
	float3 dist = maxCorner - minCorner;
	if (dist.x != 0)
		offset.x /= dist.x;
	if (dist.y != 0)
		offset.y /= dist.y;
	if (dist.z != 0)
		offset.x /= dist.z;
	return offset;
}

float3 BoundingBox::getCnetroid() const
{
	return 0.5f * (maxCorner + minCorner);
}
