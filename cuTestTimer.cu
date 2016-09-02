#include "functions.h"

StopWatchInterface *timer = NULL;

int initCUDA()
{
	int result = 0;
	result = findCudaDevice(0, NULL);
	return result;
}
void cleanCUDA()
{
	cudaDeviceReset();
}

///use to confirm myTimer is good for CUDA timing
bool testTimer1(cv::Mat &image)
{
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	if (image.type() != CV_8UC3)
	{
		return false;
	}
	cv::Mat dst_image(image.rows, image.cols, CV_8UC3);

	sdkStopTimer(&timer);
	//sdkResetTimer(&timer);
	printf("allocate host dst memory time elpse %3.1f ms\n", sdkGetTimerValue(&timer));
	sdkStartTimer(&timer);

	uchar *devPtr;
	size_t size = image.cols*image.rows*sizeof(char)* 3;
	
	checkCudaErrors(cudaMalloc((void**) &devPtr, size));
	sdkStopTimer(&timer);
	printf("allocate device dst memory time elpse %3.1f ms\n", sdkGetTimerValue(&timer));
	sdkStartTimer(&timer);


	checkCudaErrors(cudaMemcpy(devPtr, image.data, size, cudaMemcpyHostToDevice));

	sdkStopTimer(&timer);
	printf("cudaMemcpyHostToDevice time elpse %3.1f ms\n", sdkGetTimerValue(&timer));
	sdkStartTimer(&timer);


	checkCudaErrors(cudaMemcpy(dst_image.data,devPtr, size, cudaMemcpyDeviceToHost));

	sdkStopTimer(&timer);
	printf("cudaMemcpyDeviceToHost time elpse %3.1f ms\n", sdkGetTimerValue(&timer));
	sdkStartTimer(&timer);
	
	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timer);
	printf("cudaDeviceSynchronize time elpse %3.1f ms\n", sdkGetTimerValue(&timer));
	sdkStartTimer(&timer);


	checkCudaErrors(cudaFree(devPtr));
	sdkStopTimer(&timer);
	printf("cudaFree time elpse %3.1f ms\n", sdkGetTimerValue(&timer));
	sdkStartTimer(&timer);

	checkCudaErrors(cudaGetLastError());
	sdkStopTimer(&timer);
	printf("cudaGetLastError time elpse %3.1f ms\n", sdkGetTimerValue(&timer));
	
	
	sdkDeleteTimer(&timer);


	cv::namedWindow("dst_image_Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("dst_image_Display window", dst_image);                   // Show our image inside it.

	cv::waitKey(0);                                          // Wait for a keystroke in the window
	return true;
}

bool testTimer2(cv::Mat &image)
{
	
	if (image.type() != CV_8UC3)
	{
		return false;
	}
	cv::Mat dst_image;

	TIMED("malloc host dst memory")
	{
		dst_image.create(image.rows, image.cols, CV_8UC3);
	}

	

	uchar *devPtr;
	size_t size = image.cols*image.rows*sizeof(char)* 3;
	TIMED("malloc device dst memory")
	{
		checkCudaErrors(cudaMalloc((void**)&devPtr, size));
	}


	TIMED("cudaMemcpyHostToDevice")
	{
		checkCudaErrors(cudaMemcpy(devPtr, image.data, size, cudaMemcpyHostToDevice));
	}



	TIMED("cudaMemcpyDeviceToHost")
	{
		checkCudaErrors(cudaMemcpy(dst_image.data, devPtr, size, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaDeviceSynchronize());
	}


	TIMED("cudaDeviceSynchronize")
	{
		
	}

	TIMED("cudaCudaFree")
	{
		checkCudaErrors(cudaFree(devPtr));
	}
	
	TIMED("cudaGetLastError")
	{
		checkCudaErrors(cudaGetLastError());
	}



	cv::namedWindow("dst_image_Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("dst_image_Display window", dst_image);                   // Show our image inside it.

	cv::waitKey(0);                                          // Wait for a keystroke in the window
	return true;
}

