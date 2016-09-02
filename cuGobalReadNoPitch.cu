#include "functions.h"

__global__ void copyImageNoPitch(uchar * src, uchar * dst, uint imageH, uint imageW, uint channel)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < imageW) && (y < imageH))
	{
		if (channel == 3)
		{
			uchar* ptr_src = src + channel*(y*imageW + x);
			uchar* ptr_dst = dst + channel*(y*imageW + x);
			ptr_dst[0] = ptr_src[0];
			ptr_dst[1] = ptr_src[1];
			ptr_dst[2] = ptr_src[2];
		}
		else///channel == 4
		{
			uchar4* ptr_src = (uchar4*)(src) + (y*imageW + x);
			uchar4* ptr_dst = (uchar4*)(dst) + (y*imageW + x);
			ptr_dst[0] = ptr_src[0];
		}
	}
}

bool testGlobalReadNoPitch(cv::Mat &image, std::string  titile)
{

	cv::Mat dst_image;
	uchar *devPtr_src, *devPtr_dst;
	size_t size = image.cols*image.rows*sizeof(uchar)* image.channels();

	int width = image.cols;
	int height = image.rows;
	int channel = image.channels();

	if (image.type() == CV_8UC3)
	{
		TIMED("malloc host dst memory")
		{
			dst_image.create(image.rows, image.cols, CV_8UC3);
		}
	}
	else if (image.type() == CV_8UC4)
	{
		TIMED("malloc host dst memory")
		{
			dst_image.create(image.rows, image.cols, CV_8UC4);
		}
	}
	

	



	
	TIMED("malloc device dst memory")
	{
		checkCudaErrors(cudaMalloc((void**)&devPtr_src, size));
		checkCudaErrors(cudaMalloc((void**)&devPtr_dst, size));
	}

	
	TIMED("cudaMemcpyHostToDevice")
	{
		checkCudaErrors(cudaMemcpy(devPtr_src, image.data, size, cudaMemcpyHostToDevice));
	}

	

	dim3 blockSize(16, 16, 1);
	dim3 gridSize(((uint)width + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);

	TIMED("copyImage")
	{
		for (int i = 0; i < 10000; i++)
		{
			copyImageNoPitch <<<gridSize, blockSize >>>(devPtr_src, devPtr_dst,  (uint)height ,(uint)width, (uint)channel);
			checkCudaErrors(cudaDeviceSynchronize());
		}
		
	}

	

	TIMED("cudaMemcpyDeviceToHost")
	{
		checkCudaErrors(cudaMemcpy(dst_image.data, devPtr_dst, size, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaDeviceSynchronize());
		/*
		for (int i = 0; i < height; i++)
		{
			uchar* data = dst_image.ptr<uchar>(i);
			for (int j = 0; j < width; j++)
			{
				printf("%d,%d,%u,%u,%u\n", i, j, data[3 * j], data[3 * j+1], data[3 * j+2]);
			}
			printf("\n\n");
		}
		*/
	}


	TIMED("cudaDeviceSynchronize")
	{
		checkCudaErrors(cudaDeviceSynchronize());
	}

	TIMED("cudaCudaFree")
	{
		checkCudaErrors(cudaFree(devPtr_src));
		checkCudaErrors(cudaFree(devPtr_dst));

	}

	TIMED("cudaGetLastError")
	{
		checkCudaErrors(cudaGetLastError());
	}



	//cv::namedWindow(titile, cv::WINDOW_AUTOSIZE);// Create a window for display.
	//cv::imshow(titile, dst_image);                   // Show our image inside it.
	cv::imwrite("e:/tmp/" + titile + ".jpg", dst_image);
	//cv::waitKey(0);                                          // Wait for a keystroke in the window
	return true;
}