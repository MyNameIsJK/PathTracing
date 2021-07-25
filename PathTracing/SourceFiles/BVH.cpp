#include "BVH.h"
#include <algorithm>
#include <cassert>

BVH::BVH(std::vector<Triangle*>&p, int maxPrimInNode, SplitMethod spm):
	maxPrimitivesInNode(min(255,maxPrimInNode)),splitMethod(spm)
{
	if (p.empty())
		return;
	primitives.reserve(p.size());
	for (int i = 0; i < p.size(); i++)
		primitives.push_back({ p[i],i });
	root = recursiveBuild(primitives);
}

BVHBuildNode* BVH::recursiveBuild(std::vector<SortTriangle>& p)
{
	BVHBuildNode* node = new BVHBuildNode();

	BoundingBox aabb;
	aabb = p[0].triangle->getBoundingBox();

	if (p.size() == 1)
	{
		node->AABB = aabb;
		node->triangle = p[0];
		node->leftChild = nullptr;
		node->rightChild = nullptr;
		node->area = p[0].triangle->getArea();
		return node;
	}

	for (int i = 1; i < p.size(); i++)
		aabb = aabb || p[i].triangle->getBoundingBox();

	if (p.size() == 2)
	{
		node->leftChild = recursiveBuild(std::vector<SortTriangle>{ p[0] });
		node->rightChild = recursiveBuild(std::vector<SortTriangle>{p[1]});
		node->AABB = node->leftChild->AABB || node->rightChild->AABB;
		node->area = node->leftChild->area + node->rightChild->area;
		return node;
	}

	BoundingBox centroidAABB;
	centroidAABB = BoundingBox(p[0].triangle->getBoundingBox().getCnetroid(), p[0].triangle->getBoundingBox().getCnetroid());
	for (int i = 1; i < p.size(); i++)
		centroidAABB = centroidAABB || p[i].triangle->getBoundingBox().getCnetroid();
	Axis longestAxis = centroidAABB.calculateLongestAxis();
	switch (longestAxis)
	{
	case X:
		std::sort(p.begin(), p.end(), [](auto f1, auto f2) {
			return f1.triangle->getBoundingBox().getCnetroid().x <
				f2.triangle->getBoundingBox().getCnetroid().x;
			});
		break;
	case Y:
		std::sort(p.begin(), p.end(), [](auto f1, auto f2) {
			return f1.triangle->getBoundingBox().getCnetroid().y <
				f2.triangle->getBoundingBox().getCnetroid().y;
			});
		break;
	case Z:
		std::sort(p.begin(), p.end(), [](auto f1, auto f2) {
			return f1.triangle->getBoundingBox().getCnetroid().z <
				f2.triangle->getBoundingBox().getCnetroid().z;
			});
		break;
	}
	auto begining = p.begin();
	auto ending = p.end();
	auto middle = begining + p.size() / 2;

	auto leftPrimive = std::vector<SortTriangle>(begining, middle);
	auto rightPrimitive = std::vector<SortTriangle>(middle, ending);

	assert(p.size() == (leftPrimive.size() + rightPrimitive.size()));
	node->leftChild = recursiveBuild(leftPrimive);
	node->rightChild = recursiveBuild(rightPrimitive);

	node->AABB = node->leftChild->AABB || node->rightChild->AABB;
	node->area = node->leftChild->area + node->rightChild->area;
	return node;
}
BoundingBox unionBox(const BoundingBox& a, const BoundingBox& b)
{
	return BoundingBox(fmaxf(a.maxCorner, b.maxCorner),
		fminf(a.minCorner, b.minCorner)
	);
}
void BVH::dfsBVH(std::vector<BVHArrNode>& bvhArr, BVHBuildNode* root, std::vector<Triangle*>p)
{
	BVHArrNode arrNode;
	int vecIndex = bvhArr.size();
	bvhArr.push_back(arrNode);
	if (root->leftChild == nullptr && root->rightChild == nullptr)
	{
		bvhArr[vecIndex].triangleIndex = root->triangle.index;
		bvhArr[vecIndex].AABB = p[bvhArr[vecIndex].triangleIndex]->getBoundingBox();
		return;
	}
	else
	{
		dfsBVH(bvhArr, root->leftChild, p);
		bvhArr[vecIndex].rightChildIndex = bvhArr.size();
		dfsBVH(bvhArr, root->rightChild, p);
		bvhArr[vecIndex].AABB = unionBox(bvhArr[vecIndex + 1].AABB , bvhArr[bvhArr[vecIndex].rightChildIndex].AABB);
	}
}
