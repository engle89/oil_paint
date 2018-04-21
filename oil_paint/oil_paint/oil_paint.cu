#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "cutil_math.h"

uchar4* d_inputImageRGBA__;
uchar4* d_outputImageRGBA__;

__global__ void oil_effect(const uchar4* const inputImage, 
	                       uchar4* const outputImage,
	                       int intensityLevels,
	                       int numRows,
	                       int numCols,
	                       int radius)
{
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	if (col >= numCols || row >= numRows)
		return;
	int intensitycount[256] = {};
	float averageR[256] = {};
	float averageG[256] = {};
	float averageB[256] = {};
	for (int filter_r = -radius ; filter_r <= radius; ++filter_r)
	{
		for (int filter_c = -radius; filter_c <= radius; ++filter_c)
		{
			int image_r = min(max(row + filter_r, 0), static_cast<int>(numRows - 1)); //boundary handle
			int image_c = min(max(col + filter_c, 0), static_cast<int>(numCols - 1)); //boundary handle

			int temp = image_r * numCols + image_c;
			float r = static_cast<float>(inputImage[temp].x);
			float g = static_cast<float>(inputImage[temp].y);
			float b = static_cast<float>(inputImage[temp].z);
			int curIntensity = (int)(((r+g+b)/3) *intensityLevels )/ 255.0f;
			if (curIntensity > 255)
				curIntensity = 255;
			intensitycount[curIntensity]++;
			averageR[curIntensity] += r;
			averageG[curIntensity] += g;
			averageB[curIntensity] += b;
		}
	}

	int curMax = 0;
	int maxIndex = 0;
	for (int i = 0; i <= 255; i++)
	{
		if (intensitycount[i] > curMax)
		{
			curMax = intensitycount[i];
			maxIndex = i;
		}
	}
	
	int temp2 = row * numCols + col;
	outputImage[temp2].x = averageR[maxIndex] / curMax;
	outputImage[temp2].y = averageG[maxIndex] / curMax;
	outputImage[temp2].z = averageB[maxIndex] / curMax;
}

uchar4* oil(uchar4* d_in, unsigned char* d_intensity, size_t numRows, size_t numCols, int radius)
{
	const dim3 blockSize(16, 16, 1);
	int a = numCols / blockSize.x, b = numRows / blockSize.y;
	const dim3 gridSize(a + 1, b + 1, 1);
	const size_t numPixels = numRows * numCols;

	uchar4* d_out;
	cudaMalloc((void**)&d_out, sizeof(uchar4)*numPixels);
	cudaMemset(d_out, 0, numPixels * sizeof(uchar4));

	d_inputImageRGBA__ = d_in;
	d_outputImageRGBA__ = d_out;

	//oil paint
	oil_effect << <gridSize, blockSize >> > (d_in, d_out, 20, numRows, numCols, radius);
	cudaDeviceSynchronize();

	//output
	uchar4* h_out;
	h_out = (uchar4*)malloc(sizeof(uchar4)*numPixels);
	cudaMemcpy(h_out, d_out, sizeof(uchar4)*numPixels, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_inputImageRGBA__);
	cudaFree(d_outputImageRGBA__);
	
	return h_out;
}