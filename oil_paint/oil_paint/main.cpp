#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "utility.h"
#include "oil_paint.h"

using namespace cv;
using namespace std;

size_t numRows, numCols;
uchar4* d_in;
unsigned char* d_intensity;
void load_image_in_GPU(string filename)
{
	uchar4 *h_image;
	unsigned char* h_intensity;
	loadImageRGBA(filename, &h_image, &numRows, &numCols, &h_intensity);
	cudaMalloc((void**)&d_in, numRows*numCols * sizeof(uchar4));
	cudaMemcpy(d_in, h_image, numRows*numCols * sizeof(uchar4), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_intensity, numRows*numCols * sizeof(unsigned char));
	cudaMemcpy(d_intensity, h_intensity, numRows*numCols * sizeof(unsigned char), cudaMemcpyHostToDevice);
	free(h_image);
	free(h_intensity);
}

void load_frame_in_GPU(cv::Mat filename)
{
	uchar4 *h_image;
	unsigned char* h_intensity;
	loadImageRGBA2(filename, &h_image, &numRows, &numCols, &h_intensity);
	cudaMalloc((void**)&d_in, numRows*numCols * sizeof(uchar4));
	cudaMemcpy(d_in, h_image, numRows*numCols * sizeof(uchar4), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_intensity, numRows*numCols * sizeof(unsigned char));
	cudaMemcpy(d_intensity, h_intensity, numRows*numCols * sizeof(unsigned char), cudaMemcpyHostToDevice);
	free(h_image);
	free(h_intensity);
}

unsigned char* createImageBuffer(unsigned int bytes, unsigned char **devicePtr)
{
	unsigned char *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}

int main()
{
	/*
	load_image_in_GPU("D:/engle/image processing/blur/pic.png");
	uchar4* h_out = NULL;

	h_out = oil(d_in, d_intensity, numRows, numCols, 8);
	cudaFree(d_in);
	string outputfile = "D:/engle/image processing/blur/test.png";
	if (h_out != NULL)
		saveImageRGBA(h_out, outputfile, numRows, numCols);
	*/
	cv::VideoCapture camera(0);
	if (!camera.isOpened())
		return -1;

	cv::namedWindow("Source");

	while (1)
	{
		cv::Mat frame;
		bool bSuccess = camera.read(frame);
		if (bSuccess == false)
		{
			cout << "Video camera is disconnected" << endl;
			cin.get(); //Wait for any key press
			break;
		}
		load_frame_in_GPU(frame);
		uchar4* h_out = NULL;
		h_out = oil(d_in, d_intensity, numRows, numCols, 5);
		cudaFree(d_in);
		Mat outputfile;
		if (h_out != NULL)
			saveImageRGBA2(h_out, outputfile, numRows, numCols);
		cv::imshow("Source", frame);
		cv::imshow("Oil", outputfile);

		if (waitKey(10) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}
	}

	return 0;
}