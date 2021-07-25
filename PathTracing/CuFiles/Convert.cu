#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_math.h>

__device__ uchar4 flt3ToUchar4(const float3& flt3, bool isFltNormlized)
{
	uchar4 u4;
	if (isFltNormlized)
		u4 = make_uchar4(flt3.x * 255.0f, flt3.y * 255.0f, flt3.z * 255.0f, 255);
	else
		u4 = make_uchar4(flt3.x, flt3.y, flt3.z, 255);
	return u4;
}
__global__ void imgFlt3ToUchar4(float3* src, uchar4* des, int2 imgSize)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < imgSize.x && y < imgSize.y)
	{
		int idx = y * imgSize.x + x;
		des[idx] = flt3ToUchar4(src[idx], true);
	}
}
extern "C" void convertImg(float3 * src, uchar4 * des, int2 imgSize, dim3 blockSize, dim3 gridSize)
{
	imgFlt3ToUchar4 << <gridSize, blockSize >> > (src, des, imgSize);
}