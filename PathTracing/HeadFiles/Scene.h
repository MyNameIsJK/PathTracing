#pragma once
#ifndef SCENE_H
#define SCENE_H

#include "Triangle.h"
#include "Light.h"
#include <iostream>
#include "BVH.h"
#include "Renderer.h"

class Scene
{
private:
	int2 resolution=make_int2(1280,960);
	double fov = 35;
	float3 backGroundColor=make_float3(0.235294f, 0.67451f, 0.843137f);
	int maxDepth = 1;
	float RussianRoulette = 0.8f;

	std::vector<Triangle>triangles;
	std::vector<Light>lights;
	std::vector<BVHArrNode>bvhArr;
	std::vector<uint>Source;

	int numTriangles;
	int numLights;
	int numBvhArr;
	int numSource;

	uint* sourcesDev;
	BVHArrNode* bvhArrDev;
	Triangle* trianglesDev;
	Light* lightsDev;
	float* preSumSourceAreaDev;

	dim3 blockSize;
	dim3 gridSize;
	uchar4* imgDev;
	uchar4* imgHost;
	Gbuffer* gbufferDev;
	RealTimeParam* rtpHost;
	RealTimeParam* rtpDev;
	RenderParam* renderParamDev;

	int numFilterLevel = 5;
	float sigmaN = 0.5f;
	float sigmaL = 0.5f;
	float sigmaX = 0.2f;
	int numPixel;
	float3* imgFlt1;
	float3* imgFlt2;

	cudaArray_t cudaSceneImageArray;
	cudaGraphicsResource_t cudaTextureResource;
public:
	Scene(int width, int height);
	~Scene();
	void add(MeshTriangle* meshTriangle);
	void add(Light*light);
	void printInfo();
	void buildBVH();
	void uploadData();
	void registerOpenGL(unsigned int texture);
	void draw();
};

#endif