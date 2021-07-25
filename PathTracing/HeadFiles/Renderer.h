#pragma once
#ifndef RENDERER_H
#define RENDERER_H
#include "Triangle.h"
#include "BVH.h"
struct RenderParam
{
	int numTriangles;
	int numBvhArr;

	float3 backGroundColor;
	float scale;
	float3 camPos;
	float imgAspectRatio;
	int2 resolution;

	float russianRoulette;
};
struct RealTimeParam
{
	int frameNumber;
	int hashFrameNumber;
};
struct Gbuffer
{
	float3 pos;
	float3 norm;
	float3 albedo;
	int triangleId;
};
#endif