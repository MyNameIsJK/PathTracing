#include "Intersect.h"
#include "Renderer.h"
#include "Sample.h"
#define numTriangle 32
template <int value>
__device__ bool rayBVHIntersect(const Ray& ray, BVHArrNode* bvhArr, Triangle* triangles, float& tnear, int& hitIndex,int curIndex)
{
	if (!rayBoundingBoxIntersect(ray, bvhArr[curIndex].AABB, tnear))
		return false;
	if (bvhArr[curIndex].triangleIndex != -1)
	{
		if (rayTriangleIntersect(ray, triangles[bvhArr[curIndex].triangleIndex], tnear))
		{
			hitIndex = bvhArr[curIndex].triangleIndex;
			return true;
		}
		else
			return false;
	}
	bool iLeft = false;
	bool iRight = false;
	float tLeft;
	float tRight;
	int hitLeft;
	int hitRight;
	iLeft = rayBVHIntersect<value+1>(ray, bvhArr, triangles, tLeft, hitLeft, curIndex + 1);
	iRight = rayBVHIntersect<value + 1>(ray, bvhArr, triangles, tRight, hitRight, bvhArr[curIndex].rightChildIndex);
	if (iLeft == false && iRight == false)
		return false;
	else if (iLeft == true && iRight == false)
	{
		hitIndex = hitLeft;
		tnear = tLeft;
	}
	else if (iLeft == false && iRight == true)
	{
		hitIndex = hitRight;
		tnear = tRight;
	}
	else
	{
		if (tLeft < tRight)
		{
			hitIndex = hitLeft;
			tnear = tLeft;
		}
		else
		{
			hitIndex = hitRight;
			tnear = tRight;
		}
	}
	return true;
}
template<>
__device__ bool rayBVHIntersect<MAX_BVH_DEPTH>(const Ray& ray, BVHArrNode* bvhArr, Triangle* triangles, float& tnear, int& hitIndex, int curIndex)
{
	return false;
}

__device__ bool trace(const Ray& ray, Triangle* triangles, float& tnear, int& hitIndex)
{
	float tmpt;
	tnear = FLT_MAX;
	hitIndex = -1;
	for (int i = 0; i < numTriangle; i++)
	{
		if (rayTriangleIntersect(ray, triangles[i], tmpt))
		{
			if (tmpt < tnear)
			{
				tnear = tmpt;
				hitIndex = i;
			}
		}
	}
	if (hitIndex >= 0)
		return true;
	return false;
}
template <int value>
__device__ float3 castRay(Triangle* triangles, BVHArrNode* bvhArr, float3 backGroundColor, Ray ray, int numTri, curandState& randState,
	uint* sources, uint numSources, float* preSumArea, RenderParam* renderParam, Gbuffer& gbuffer)
{
	int hitIndex;
	float tnear;
	//if (trace(ray, triangles, tnear, hitIndex))
	if (value == 0)
		gbuffer.triangleId = -1;
	if (rayBVHIntersect<0>(ray, bvhArr, triangles, tnear, hitIndex,0))
	{
		if (triangles[hitIndex].hasEmission())
			return triangles[hitIndex].material.emission;
		float3 color = make_float3(0.0f);
		float3 hitPos = ray.origin + tnear * ray.direction;
		float3 hitNormal = triangles[hitIndex].normal;
		if (value == 0)
		{
			gbuffer.pos = hitPos;
			gbuffer.norm = hitNormal;
			gbuffer.triangleId = hitIndex;
			gbuffer.albedo = triangles[hitIndex].material.kd;
		}
		float3 lightPos;
		float3 lightNormal;
		float lightPdf;
		sampleLight(lightPos, lightNormal, lightPdf, sources, numSources, randState, preSumArea, triangles);
		///*
		float3 objToLight = lightPos - hitPos;
		float3 objToLightDir = normalize(objToLight);
		float distToLight2 = dot(objToLight, objToLight);
		Ray shadowRay(hitPos, objToLightDir);
		float shadowTnear;
		int shadowHitIndex;
		//trace(shadowRay, triangles, shadowTnear, shadowHitIndex);
		rayBVHIntersect<0>(shadowRay, bvhArr, triangles, shadowTnear, shadowHitIndex, 0);
		// shadowRay击中了光源
		if (dot(shadowTnear * shadowRay.direction, shadowTnear * shadowRay.direction) - distToLight2 > -EPSILON)
		{
			//return make_float3(dot(objToLightDir, hitNormal));
			color = triangles[shadowHitIndex].material.emission / distToLight2 / lightPdf * triangles[hitIndex].material.eval(ray.direction, objToLightDir, hitNormal) *
				dot(objToLightDir, hitNormal) * dot(-objToLightDir, triangles[shadowHitIndex].normal);
			color = fminf(fmaxf(color, make_float3(0.0f)), make_float3(1.0f));
		}
		if (curand_uniform(&randState) > 0.8f)//renderParam->russianRoulette)
		{
			return color;
		}
		float3 nextRayDir = normalize(sampleDiffuseMaterial(hitNormal, randState));
		Ray nextRay(hitPos, nextRayDir);
		int nextHitIndex;
		float nextHitTnear;
		//bool isInter = trace(nextRay, triangles, nextHitTnear, nextHitIndex);
		bool isInter = rayBVHIntersect<0>(nextRay, bvhArr, triangles, nextHitTnear, nextHitIndex, 0);
		float3 nextColor = make_float3(0.0f);
		if (isInter)
		{
			if (!triangles[nextHitIndex].hasEmission())
			{
				float pdf;
				if (dot(nextRayDir, hitNormal) > 0.0f)
					pdf = 0.5f / M_PI;
				else
					pdf = 0.0f;
				nextColor = castRay<value + 1>(triangles, bvhArr, backGroundColor, nextRay, numTri, randState, sources, numSources, preSumArea, renderParam, gbuffer) *
					triangles[hitIndex].material.eval(ray.direction, nextRayDir, hitNormal) * dot(nextRayDir, hitNormal) / pdf / renderParam->russianRoulette;
			}
		}
		nextColor = fminf(fmaxf(nextColor, make_float3(0.0f)), make_float3(1.0f));
		//*/
		return color + nextColor;
	}
	return backGroundColor;
}
template<>
__device__ float3 castRay<MAX_RAY_DEPTH>(Triangle* triangles, BVHArrNode* bvhArr, float3 backGroundColor, Ray ray, int numTri, curandState& randState,
	uint* sources, uint numSources, float* preSumArea, RenderParam* renderParam, Gbuffer& gbuffer)
{
	return backGroundColor;
}
__global__ void redererKernel(Triangle* triangles,BVHArrNode* bvhArr,
	float3* frameBuffer, RenderParam* renderParam, RealTimeParam* realTimeParam,
	uint* sources, uint numSources, float* preSumArea, Gbuffer*gbuffer)
{
	uint idxx = threadIdx.x + blockIdx.x * blockDim.x;
	uint idxy = threadIdx.y + blockIdx.y * blockDim.y;
	if (idxx < renderParam->resolution.x && idxy < renderParam->resolution.y)
	{
		float x = (float)(2 * idxx + 1) / (float)(renderParam->resolution.x) - 1.0f;
		x = x * renderParam->imgAspectRatio * renderParam->scale;
		float y = (1.0f - (float)(2 * idxy + 1) / (float)(renderParam->resolution.y)) * renderParam->scale;
		float3 dir = normalize(make_float3(-x, y, 1));
		// create random number generator and initialise with hashed frame number, see RichieSams blogspot
		curandState randState; // state of the random number generator, to prevent repetition
		int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
		curand_init(realTimeParam->hashFrameNumber + threadId, 0, 0, &randState);
		Ray originRay(renderParam->camPos, dir);
		Ray tmpRay;
		float3 color = make_float3(0.0f);
		int spp = SPP;
		//if (realTimeParam->frameNumber == 0)
		//	spp = 16;
		for (int i = 0; i < spp; i++)
		{
			float jitterValueX = curand_uniform(&randState) - 0.5;
			float jitterValueY = curand_uniform(&randState) - 0.5;
			jitterValueX /= (float)(renderParam->resolution.x);
			jitterValueX = jitterValueX * renderParam->imgAspectRatio * renderParam->scale;
			jitterValueY /= (float)(renderParam->resolution.y);
			jitterValueY = jitterValueY * renderParam->scale;
			tmpRay = originRay;
			tmpRay.direction.x += jitterValueX;
			tmpRay.direction.y += jitterValueY;
			color += castRay<0>(triangles, bvhArr,
				renderParam->backGroundColor, tmpRay, renderParam->numBvhArr,
				randState, sources, numSources, preSumArea, renderParam, gbuffer[idxy * renderParam->resolution.x + idxx]) / (float)spp;
		}
		color = fminf(fmaxf(color,make_float3(0.0f)), make_float3(1.0f));
		color = make_float3(pow(color.x, 0.6f), pow(color.y, 0.6f), pow(color.z, 0.6f));
		frameBuffer[idxy * renderParam->resolution.x + idxx] = color;
			//make_uchar4(uchar(color.x * 255.0f), uchar(color.y * 255.0f), uchar(color.z * 255.0f), 255);
	}
}
extern "C" void redererCuda(Triangle * triangles,BVHArrNode * bvhArr, float3 * frameBuffer,
	RenderParam*renderParam, RealTimeParam*realTimeParam, 
	uint * sources, uint numSources, float* preSumArea,Gbuffer*gbuffer,
	dim3 blockSize, dim3 gridSize)
{
	redererKernel << <gridSize, blockSize >> > (triangles, bvhArr,
		frameBuffer, renderParam, realTimeParam, sources, numSources, preSumArea, gbuffer);
}