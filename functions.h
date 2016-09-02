#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

// includes, cuda
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <vector_types.h>

#include <helper_functions.h>
#include "myTimer.h"

typedef unsigned int  uint;
typedef unsigned char uchar;

int initCUDA();
void cleanCUDA();

bool testTimer1(cv::Mat &image);
bool testTimer2(cv::Mat &image);

bool testGlobalReadNoPitch(cv::Mat &image,std::string  titile);
bool testGlobalReadPitch(cv::Mat &image, std::string  titile);

bool testTexReadNoPitch(cv::Mat &image, std::string  titile);
bool testTexReadPitch(cv::Mat &image, std::string  titile);

bool testcuArrayRead(cv::Mat &image, std::string  titile);