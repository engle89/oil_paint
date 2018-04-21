#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

void loadImageRGBA(string& filename, uchar4** imagePtr, size_t* numRows, size_t* numCols, unsigned char** intensity)
{
	Mat image = imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
	{
		cerr << "Failed to load image: " << filename << endl;
		exit(1);
	}
	if (image.channels() != 3)
	{
		cerr << "Image must be color!" << endl;
		exit(1);
	}
	if (!image.isContinuous()) {
		cerr << "Image isn't continuous!" << endl;
		exit(1);
	}

	Mat imageRGBA;
	cvtColor(image, imageRGBA, CV_BGR2RGBA);

	*imagePtr = new uchar4[image.rows*image.cols];
	unsigned char* cvPtr = imageRGBA.ptr<unsigned char>(0);
	for (size_t i = 0; i < image.rows*image.cols; ++i) {
		(*imagePtr)[i].x = cvPtr[4 * i + 0];
		(*imagePtr)[i].y = cvPtr[4 * i + 1];
		(*imagePtr)[i].z = cvPtr[4 * i + 2];
		(*imagePtr)[i].w = cvPtr[4 * i + 3];
	}

	*numRows = image.rows;
	*numCols = image.cols;

	//intensity
	*intensity = new unsigned char[image.rows*image.cols];
	for (size_t i = 0; i < image.rows; ++i)
		for (size_t j = 0; j < image.cols; j++)
			(*intensity)[i*j] = image.at<uchar>(i, j);
}

void loadImageRGBA2(Mat& filename, uchar4** imagePtr, size_t* numRows, size_t* numCols, unsigned char** intensity)
{
	if (filename.empty())
	{
		cerr << "Failed to load image: " << filename << endl;
		exit(1);
	}
	if (filename.channels() != 3)
	{
		cerr << "Image must be color!" << endl;
		exit(1);
	}
	if (!filename.isContinuous()) {
		cerr << "Image isn't continuous!" << endl;
		exit(1);
	}

	Mat imageRGBA;
	cvtColor(filename, imageRGBA, CV_BGR2RGBA);

	*imagePtr = new uchar4[filename.rows*filename.cols];
	unsigned char* cvPtr = imageRGBA.ptr<unsigned char>(0);
	for (size_t i = 0; i < filename.rows*filename.cols; ++i) {
		(*imagePtr)[i].x = cvPtr[4 * i + 0];
		(*imagePtr)[i].y = cvPtr[4 * i + 1];
		(*imagePtr)[i].z = cvPtr[4 * i + 2];
		(*imagePtr)[i].w = cvPtr[4 * i + 3];
	}

	*numRows = filename.rows;
	*numCols = filename.cols;

	//intensity
	*intensity = new unsigned char[filename.rows*filename.cols];
	for (size_t i = 0; i < filename.rows; ++i)
		for (size_t j = 0; j < filename.cols; j++)
			(*intensity)[i*j] = filename.at<uchar>(i, j);
}


void saveImageRGBA(uchar4* image, string& output_filename, size_t numRows, size_t numCols)
{
	int sizes[2] = { numRows, numCols };
	Mat imageRGBA(2, sizes, CV_8UC4, (void*)image);
	Mat imageOutputBGR;
	cvtColor(imageRGBA, imageOutputBGR, CV_RGBA2BGR);
	imwrite(output_filename.c_str(), imageOutputBGR);
}

void saveImageRGBA2(uchar4* image, Mat& output_filename, size_t numRows, size_t numCols)
{
	int sizes[2] = { numRows, numCols };
	Mat imageRGBA(2, sizes, CV_8UC4, (void*)image);
	cvtColor(imageRGBA, output_filename, CV_RGBA2BGR);
}

