#ifndef UTILITY_H
#define UTILITY_H

#include <string>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

void loadImageRGBA(std::string& filename, uchar4** imagePtr,size_t* numRows, size_t* numCols, unsigned char** intensity);
void loadImageRGBA2(cv::Mat &filename, uchar4** imagePtr, size_t* numRows, size_t* numCols, unsigned char** intensity);
void saveImageRGBA(uchar4* image, std::string& output_filename, size_t numRows, size_t numCols);
void saveImageRGBA2(uchar4* image, cv::Mat& output_filename, size_t numRows, size_t numCols);
#endif
