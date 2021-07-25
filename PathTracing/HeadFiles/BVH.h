#pragma once
#ifndef BVH_H
#define BVH_H

#include "BoundingBox.h"
#include "Triangle.h"

enum SplitMethod
{
	NAIVE,
	SAH
};
//为了防止BVH数组的中间节点也需要申请一个三角形空间
struct SortTriangle
{
	Triangle* triangle;
	int index;
	SortTriangle(Triangle* t, int i)
	{
		triangle = t;
		index = i;
	}
	SortTriangle() {}
};
struct BVHBuildNode
{
	BoundingBox AABB;
	BVHBuildNode* leftChild;
	BVHBuildNode* rightChild;

	SortTriangle triangle;

	float area;

	Axis splitAxis = X;

	BVHBuildNode()
	{
		AABB = BoundingBox();
		leftChild = rightChild = nullptr;
	}
};
struct BVHArrNode
{
	BoundingBox AABB;
	int triangleIndex = -1;
	int rightChildIndex = -1;
};
class BVH
{
public:
	const int maxPrimitivesInNode;
	const SplitMethod splitMethod;
	std::vector<SortTriangle>primitives;
	BVHBuildNode* root;
public:
	BVH(std::vector<Triangle*>&p, int maxPrimInNode = 1, SplitMethod spm = NAIVE);
	BVHBuildNode* recursiveBuild(std::vector<SortTriangle>&p);
	void dfsBVH(std::vector<BVHArrNode>& bvhArr, BVHBuildNode* root, std::vector<Triangle*>p);
};

#endif