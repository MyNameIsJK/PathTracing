#include "Renderer.h"
#include "Shader.h"
#include <helper_cuda.h>

float minColorAlpha = 0.8f;
float minMomentAlpha = 0.8f;
// luminance, luminance*luminance
float2* momentPrev;
// color 
float3* colorPrev;
// dont use it now
glm::mat4 transformPrev;
// frame number
int* lengthPrev;
// gbuffer
Gbuffer* gbufferPrev;
// variance compute by Var=E(X2)-E(X)2
float* variance;

float2* momentAcc;
float3* colorAcc;
// store temp data
float3* tmpDev[2];

extern "C" void denoiseInit(int numPixel)
{
	checkCudaErrors(cudaMalloc((void**)&tmpDev[0], numPixel * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&tmpDev[1], numPixel * sizeof(float3)));

	checkCudaErrors(cudaMalloc((void**)&lengthPrev, numPixel * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&gbufferPrev, numPixel * sizeof(Gbuffer)));
	checkCudaErrors(cudaMalloc((void**)&momentPrev, numPixel * sizeof(float2)));
	checkCudaErrors(cudaMalloc((void**)&colorPrev, numPixel * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&momentAcc, numPixel * sizeof(float2)));
	checkCudaErrors(cudaMalloc((void**)&colorAcc, numPixel * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&variance, numPixel * sizeof(float)));

	checkCudaErrors(cudaMemset(lengthPrev,0, numPixel * sizeof(int)));
	checkCudaErrors(cudaMemset(gbufferPrev, 0, numPixel * sizeof(Gbuffer)));
	checkCudaErrors(cudaMemset(momentPrev, 0, numPixel * sizeof(float2)));
	checkCudaErrors(cudaMemset(colorPrev, 0, numPixel * sizeof(float3)));
	checkCudaErrors(cudaMemset(momentAcc, 0, numPixel * sizeof(float2)));
	checkCudaErrors(cudaMemset(colorAcc, 0, numPixel * sizeof(float3)));
	checkCudaErrors(cudaMemset(variance, 0, numPixel * sizeof(float)));
}

extern "C" void denoiseFree()
{
	checkCudaErrors(cudaFree(tmpDev[0]));
	checkCudaErrors(cudaFree(tmpDev[1]));
	checkCudaErrors(cudaFree(lengthPrev));
	checkCudaErrors(cudaFree(gbufferPrev));
	checkCudaErrors(cudaFree(momentPrev));
	checkCudaErrors(cudaFree(colorPrev));
	checkCudaErrors(cudaFree(momentAcc));
	checkCudaErrors(cudaFree(colorAcc));
	checkCudaErrors(cudaFree(variance));
}

__global__ void ATrousFilter(float3*inputImg,float3*outputImg, int2 imgSize,
	float*variance, Gbuffer*gbuffer, int level, bool isLastFilter, 
	float sigmaC,float sigmaN,float sigmaX, bool isBlurVar, bool addColor)
{
	// 5x5 A-Trous kernel
	float h[25] = { 1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
					1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
					3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0, 3.0 / 32.0, 3.0 / 128.0,
					1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
					1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0 };

	// 3x3 Gaussian kernel
	float gaussian[9] = { 1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
						  1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
						  1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0 };

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < imgSize.x && y < imgSize.y)
	{
		int pixelIdx = x + y * imgSize.x;
		int step = 1 << level;
		float var;
		// perform 3x3 gaussian blur on variance
		if (isBlurVar)
		{
			float sum = 0.0f;
			int2 idxStep[9] = {
				make_int2(-1, -1), make_int2(0, -1), make_int2(1, -1),
				make_int2(-1, 0),  make_int2(0, 0),  make_int2(1, 0),
				make_int2(-1, 1),  make_int2(0, 1),  make_int2(1, 1)
			};
			for (int i = 0; i < 9; i++)
			{
				int2 localIdx = make_int2(x, y) + idxStep[i];
				localIdx = clamp(localIdx, make_int2(0), imgSize);
				//if (localIdx.x >= 0 && localIdx.y >= 0 && localIdx.x < imgSize.x && localIdx.y < imgSize.y)
				//{
					sum += gaussian[i] * variance[localIdx.y * imgSize.x + localIdx.x];
				//}
				var = max(0.0f, sum);
			}
		}
		else
			var = max(0.0f, variance[pixelIdx]);

		float pixelLuminance = 0.2126f * inputImg[pixelIdx].x + 0.7152f * inputImg[pixelIdx].y + 0.0722f * inputImg[pixelIdx].z;
		float3 pixelPos = gbuffer[pixelIdx].pos;
		float3 pixelNorm = gbuffer[pixelIdx].norm;

		float3 colorSum = make_float3(0.0f);
		float varSum = 0.0f;
		float weightSum = 0.0f;
		float weightSum2 = 0.0f;

		for (int i = -2; i <= 2; i++)
		{
			for (int j = -2; j <= 2; j++)
			{
				int xq = x + step * i;
				int yq = y + step * j;
				if (xq >= 0 && xq < imgSize.x && yq >= 0 && yq < imgSize.y)
				{
					int idxq = yq * imgSize.x + xq;

					float luminanceQ= 0.2126f * inputImg[idxq].x + 0.7152f * inputImg[idxq].y + 0.0722f * inputImg[idxq].z;
					float3 posQ = gbuffer[idxq].pos;
					float3 normQ = gbuffer[idxq].norm;

					float wLuminance = expf(-fabsf(pixelLuminance - luminanceQ) / (sqrt(var) * sigmaC + 1e-6));
					float wNorm = min(1.0f, expf(-length(pixelNorm - normQ) / (sigmaN + 1e-6)));
					float wPos = min(1.0f, expf(-length(pixelPos - posQ) / (sigmaX + 1e-6)));

					// filter weight
					int k = (2 + i) + (2 + j) * 5;
					float weight = h[k] * wLuminance * wPos * wNorm;
					weightSum += weight;
					weightSum2 += weight * weight;
					colorSum += weight * inputImg[idxq];
					varSum += weight * weight * variance[idxq];

				}
			}
		}

		// update color and variance
		if (weightSum > 10e-6) 
		{
			outputImg[pixelIdx] = colorSum / weightSum;
			variance[pixelIdx] = varSum / weightSum2;
		}
		else 
			outputImg[pixelIdx] = inputImg[pixelIdx];

		//if (isLastFilter && addColor)
		//	outputImg[pixelIdx] *= gbuffer[pixelIdx].albedo;
	}
}

__global__ void BackProjection(float* variance, int* lengthPrev, float2* momentPrev, float3* colorPrev, float2* momentAcc, float3* colorAcc,
	float3* colorCur, Gbuffer* gbufferCur, Gbuffer* gbufferPrev, int2 imgSize, float colorAlphaMin, float momentAlphaMin)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < imgSize.x && y < imgSize.y)
	{
		int pixelIdx = x + y * imgSize.x;
		int frameNumber = lengthPrev[pixelIdx];
		float3 sample = colorCur[pixelIdx];
		float luminance = 0.2126 * sample.x + 0.7152 * sample.y + 0.0722 * sample.z;
		if (frameNumber > 0 && gbufferCur[pixelIdx].triangleId != -1)
		{
			// 假设不进行运动
			float colorAlpha = max(1.0f / (float)(frameNumber + 1), colorAlphaMin);
			float momentAlpha = max(1.0f / (float)(frameNumber + 1), momentAlphaMin);
			float3 colorLast = colorPrev[pixelIdx];
			float2 momentLast = momentPrev[pixelIdx];
			float lengthLast = frameNumber;
			lengthPrev[pixelIdx]++;
			// color accumulation
			colorAcc[pixelIdx] = lerp(colorCur[pixelIdx], colorLast, colorAlpha);
			//colorAcc[pixelIdx] = colorLast;
			// moment accumulation
			float2 momentCur = make_float2(luminance, luminance * luminance);
			momentAcc[pixelIdx] = lerp(momentCur, momentLast, momentAlpha);
			// calculate variance  V=E(X2)-E(X)2
			float var = momentCur.y - momentCur.x * momentCur.x;
			variance[pixelIdx] = max(var, 0.0f);
		}
		else
		{
			lengthPrev[pixelIdx]++;
			colorAcc[pixelIdx] += colorCur[pixelIdx];
			momentAcc[pixelIdx] += make_float2(luminance, luminance * luminance);
			variance[pixelIdx] = 100.0f;
		}
	}
}

extern "C" void denoise(int numPixel, float3 * inputImg, float3 * outputImg, Gbuffer * gbufferCur, dim3 gridSize, int numLevel, dim3 blockSize, int2 imgSize,
	float sigmaC, float sigmaN, float sigmaX, bool isBlurVar, bool addColor, int lastFrameNumber)
{
	BackProjection << <gridSize, blockSize >> >
		(variance, lengthPrev, momentPrev, colorPrev, momentAcc, colorAcc,
			inputImg, gbufferCur, gbufferPrev, imgSize, minColorAlpha, minMomentAlpha);

	checkCudaErrors(cudaMemcpy(colorPrev, colorAcc, numPixel * sizeof(float3), cudaMemcpyDeviceToDevice));
	
	for (int level = 1; level <= numLevel; level++)
	{
		float3* src = (level == 1) ? colorPrev : tmpDev[(level & 1)];
		float3* dst = (level == numLevel) ? outputImg : tmpDev[((level + 1) & 1)];
		ATrousFilter << <gridSize, blockSize >> >
			(src, dst, imgSize, variance, gbufferCur, level, (level == numLevel),
				sigmaC, sigmaN, sigmaX, isBlurVar, addColor);
	}
	
	checkCudaErrors(cudaMemcpy(colorPrev, outputImg, numPixel * sizeof(float3), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(gbufferPrev, gbufferCur, numPixel * sizeof(Gbuffer), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(momentPrev, momentAcc, numPixel * sizeof(float2), cudaMemcpyDeviceToDevice));
}