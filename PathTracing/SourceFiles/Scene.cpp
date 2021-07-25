#include "Scene.h"
#include <helper_cuda.h>
#include <helper_gl.h>
#include <cuda_gl_interop.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using namespace std;
unsigned int WangHash(unsigned int a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}
int iDivUp(int a, int b)
{
	int ret = a / b;
	if (a % b != 0)
		ret++;
	return ret;
}
inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }
extern "C" void denoiseInit(int numPixel);
Scene::Scene(int width, int height)
{
	resolution = make_int2(width, height);
	numPixel = width * height;
	denoiseInit(numPixel);
	imgHost = new uchar4[numPixel];
	checkCudaErrors(cudaMalloc((void**)&imgDev, numPixel * sizeof(uchar4)));
	checkCudaErrors(cudaMalloc((void**)&gbufferDev, numPixel * sizeof(Gbuffer)));
	checkCudaErrors(cudaMalloc((void**)&renderParamDev, sizeof(RenderParam)));
	checkCudaErrors(cudaMalloc((void**)&imgFlt1, numPixel * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&imgFlt2, numPixel * sizeof(float3)));
	rtpHost=new RealTimeParam();
	checkCudaErrors(cudaMalloc((void**)&rtpDev, sizeof(RealTimeParam)));

	blockSize = dim3(16, 16);
	gridSize = dim3(iDivUp(width, 16), iDivUp(height, 16));
}

void Scene::add(MeshTriangle* meshTriangle)
{
	triangles.insert(triangles.end(),
		meshTriangle->triangles.begin(), meshTriangle->triangles.end());
}

void Scene::add(Light* light)
{
	lights.push_back(*light);
}

void Scene::printInfo()
{
	cout << "now scene has " << triangles.size() << " triangles and " 
		<< lights.size() << " lights" << endl;
}

void Scene::buildBVH()
{
	std::vector<Triangle*>tmpTri;
	tmpTri.resize(triangles.size());
	for (int i = 0; i < triangles.size(); i++)
		tmpTri[i] = &(triangles[i]);
	std::vector<Triangle*>tmpTri2(tmpTri);
	BVH bvhBuild(tmpTri);
	BVHBuildNode* root = bvhBuild.root;
	bvhBuild.dfsBVH(bvhArr, root, tmpTri2);
	cout << "native BVH tree has " << bvhArr.size() << " nodes" << endl;
}
void Scene::uploadData()
{
	numTriangles = triangles.size();
	numLights = lights.size();
	numBvhArr = bvhArr.size();

	for (int i=0;i<triangles.size();i++)
		if (triangles[i].hasEmission())
			Source.push_back(i);
	float* preSum = new float[Source.size()];
	numSource = Source.size();
	if (Source.size() != 0)
	{
		preSum[0] = triangles[Source[0]].getArea();
		for (int i = 1; i < Source.size(); i++)
			preSum[i] = preSum[i - 1] + triangles[Source[i]].getArea();
		checkCudaErrors(cudaMalloc((void**)&sourcesDev, Source.size() * sizeof(uint)));
		checkCudaErrors(cudaMemcpy(sourcesDev, Source.data(), Source.size() * sizeof(uint), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&preSumSourceAreaDev, Source.size() * sizeof(float)));
		checkCudaErrors(cudaMemcpy(preSumSourceAreaDev, preSum, Source.size() * sizeof(float), cudaMemcpyHostToDevice));
	}
	delete[]preSum;

	checkCudaErrors(cudaMalloc((void**)&trianglesDev, numTriangles * sizeof(Triangle)));
	checkCudaErrors(cudaMalloc((void**)&lightsDev, numLights * sizeof(Light)));
	checkCudaErrors(cudaMalloc((void**)&bvhArrDev, numBvhArr * sizeof(BVHArrNode)));

	checkCudaErrors(cudaMemcpy(trianglesDev, triangles.data(), numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(lightsDev, lights.data(), numLights * sizeof(Light), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(bvhArrDev, bvhArr.data(), numBvhArr * sizeof(BVHArrNode), cudaMemcpyHostToDevice));

	RenderParam* rp = new RenderParam();
	rp->backGroundColor = backGroundColor;
	rp->camPos = make_float3(278, 273, -800);
	rp->imgAspectRatio = (float)(resolution.x) / (float)(resolution.y);
	rp->numBvhArr = numBvhArr;
	rp->numTriangles = numTriangles;
	rp->resolution = resolution;
	rp->scale = tan(deg2rad(fov * 0.5f));
	rp->russianRoulette = RussianRoulette;
	checkCudaErrors(cudaMemcpy(renderParamDev, rp, sizeof(RenderParam), cudaMemcpyHostToDevice));
}
void Scene::registerOpenGL(unsigned int texture)
{
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaTextureResource, texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
	cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar4>();
	checkCudaErrors(cudaMallocArray(&cudaSceneImageArray, &cuDesc, resolution.x, resolution.y));
}
extern "C" void redererCuda(Triangle * triangles, BVHArrNode * bvhArr, float3 * frameBuffer,
	RenderParam * renderParam, RealTimeParam * realTimeParam,
	uint * sources, uint numSources, float* preSumArea, Gbuffer * gbuffer,
	dim3 blockSize, dim3 gridSize);
extern "C" void denoise(int numPixel, float3 * inputImg, float3 * outputImg,
	Gbuffer * gbufferCur, dim3 gridSize, int numLevel, dim3 blockSize, int2 imgSize,
	float sigmaC, float sigmaN, float sigmaX, bool isBlurVar, bool addColor, int lastFrameNumber);
extern "C" void convertImg(float3 * src, uchar4 * des, int2 imgSize, dim3 blockSize, dim3 gridSize);
void Scene::draw()
{
	static int frameNumber = 0;
	uint hashFrame = WangHash(frameNumber);
	rtpHost->frameNumber = frameNumber;
	rtpHost->hashFrameNumber = hashFrame;
	checkCudaErrors(cudaMemcpy(rtpDev, rtpHost, sizeof(RealTimeParam), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaGraphicsMapResources(1, &cudaTextureResource, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cudaSceneImageArray, cudaTextureResource, 0, 0));
	cudaThreadSynchronize();

	redererCuda(trianglesDev, bvhArrDev, imgFlt1, renderParamDev, rtpDev,sourcesDev,numSource,preSumSourceAreaDev, gbufferDev, blockSize, gridSize);

	denoise(numPixel, imgFlt1, imgFlt2, gbufferDev, gridSize, numFilterLevel, blockSize, resolution,
		sigmaL, sigmaN, sigmaX, true, true, frameNumber);
	convertImg(imgFlt2, imgDev, resolution, blockSize, gridSize);

	checkCudaErrors(cudaMemcpy(imgHost, imgDev, resolution.x * resolution.y * sizeof(uchar4), cudaMemcpyDeviceToHost));
	stbi_write_png("./img.png", resolution.x, resolution.y, 4, imgHost,0);

	cudaThreadSynchronize();
	checkCudaErrors(cudaMemcpyToArray(cudaSceneImageArray, 0, 0, imgDev, resolution.x * resolution.y * sizeof(uchar4), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTextureResource, 0));

	frameNumber++;
}

extern "C" void denoiseFree();
Scene::~Scene()
{
	denoiseFree();
	if (!triangles.empty())triangles.clear();
	if (!lights.empty())lights.clear();
	if (!bvhArr.empty())bvhArr.clear();
	if (!Source.empty())Source.clear();

	if(sourcesDev)checkCudaErrors(cudaFree(sourcesDev));
	if (bvhArrDev)checkCudaErrors(cudaFree(bvhArrDev));
	if (trianglesDev)checkCudaErrors(cudaFree(trianglesDev));
	if (lightsDev)checkCudaErrors(cudaFree(lightsDev));
	if (preSumSourceAreaDev)checkCudaErrors(cudaFree(preSumSourceAreaDev));

	if (imgDev)checkCudaErrors(cudaFree(imgDev));
	if (imgHost)delete[]imgHost;
	if (gbufferDev)checkCudaErrors(cudaFree(gbufferDev));
	if (rtpHost)delete rtpHost;
	if (rtpDev)checkCudaErrors(cudaFree(rtpDev));
	if (renderParamDev)checkCudaErrors(cudaFree(renderParamDev));

	if (imgFlt1)checkCudaErrors(cudaFree(imgFlt1));
	if (imgFlt2)checkCudaErrors(cudaFree(imgFlt2));
}