#include "functions.h"
#include "functions.h"

//this one is wrong!!!!
__global__ void copyImagePitch(uchar * src, uchar * dst, uint imageH, uint imageW, uint channel)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < imageW) && (y < imageH))
	{	
		uchar* ptr_src = src + 3* (y * imageW + x);
		uchar* ptr_dst = dst + 3* (y * imageW + x);
		/*
		ptr_dst[0] = (y * imageW + 3 * x) + 0;
		ptr_dst[1] = (y * imageW + 3 * x) + 1;
		ptr_dst[2] = (y * imageW + 3 * x) + 2;
		*/
		ptr_dst[0] = ptr_src[0];
		ptr_dst[1] = ptr_src[1];
		ptr_dst[2] = ptr_src[2];

	}
}
__global__ void copyImagePitch(uchar * src, uchar * dst, uint imageH, uint imageW, uint pitch ,uint channel)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < imageW ) && (y < imageH))
	{
		if (channel == 3)
		{
			uchar* ptr_src = src + (y * pitch + 3 * x);
			uchar* ptr_dst = dst + (y * pitch + 3 * x);
			/*
			ptr_dst[0] = (y * pitch + 3 * x) + 0;
			ptr_dst[1] = (y * pitch + 3 * x) + 1;
			ptr_dst[2] = (y * pitch + 3 * x) + 2;
			*/
			ptr_dst[0] = ptr_src[0];
			ptr_dst[1] = ptr_src[1];
			ptr_dst[2] = ptr_src[2];
		}
		else ///channel == 4
		{
			uchar4* ptr_src = (uchar4*)(src + y * pitch) + x;
			uchar4* ptr_dst = (uchar4*)(dst + y * pitch) + x;
			ptr_dst[0] = ptr_src[0];
		}
		
	}
}

bool testGlobalReadPitch(cv::Mat &image, std::string  titile)
{
	uchar *devPtr_src, *devPtr_dst;
	size_t size_width = image.cols*sizeof(uchar)* image.channels();
	size_t size_height = image.rows;
	size_t size_pitch;
	
	cv::Mat dst_image;
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
	TIMED("malloc device dst memory,and clear it!!")
	{
		checkCudaErrors(cudaMallocPitch((void**)&devPtr_src,&size_pitch ,size_width,size_height));
		checkCudaErrors(cudaMallocPitch((void**)&devPtr_dst, &size_pitch, size_width, size_height));
		checkCudaErrors(cudaMemset2D(devPtr_dst, size_pitch, 0, size_width, size_height));
	}

	

	

	TIMED("cudaMemcpyHostToDevice")
	{
		//cudaPitchedPtr tep = make_cudaPitchedPtr(image.data, size_pitch, size_width, size_height);
		checkCudaErrors(cudaMemcpy2D(devPtr_src,size_pitch, image.data, size_width,size_width,size_height, cudaMemcpyHostToDevice));
	}
	
	int width = image.cols;
	int height = image.rows;
	int channel = image.channels();

	dim3 blockSize(16, 16, 1);
	dim3 gridSize(((uint)width + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);
	//dim3 gridSize(((uint)size_pitch/3 + 1 + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);

	TIMED("copyImage")
	{
		for (int i = 0; i < 10000; i++)
		{
			//copyImagePitch << <gridSize, blockSize >> >(devPtr_src, devPtr_dst, (uint)height, (uint)width, (uint)channel);
			copyImagePitch << <gridSize, blockSize >> >(devPtr_src, devPtr_dst, (uint)height, (uint)width, (uint)size_pitch ,(uint)channel);

			checkCudaErrors(cudaDeviceSynchronize());
		}

	}
	
	

	TIMED("cudaMemcpyDeviceToHost")
	{
		checkCudaErrors(cudaMemcpy2D(dst_image.data, size_width, devPtr_dst, size_pitch,size_width,size_height, cudaMemcpyDeviceToHost));

		////memory crashed!below
		//checkCudaErrors(cudaMemcpy(dst_image.data, devPtr_dst, size_width*size_height*3, cudaMemcpyDeviceToHost));

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
	/*
	for (int i = 0; i < size_height; i++)
	{
		uchar *ptr = dst_image.ptr<uchar>(i);
		for (int j = 0; j < size_width; j++)
		{
			printf("i=%d,j=%d,c=%d,v=%u\n", i, j / 3, j % 3, ptr[j]);
		}
	}
	*/
	
	//cv::namedWindow(titile, cv::WINDOW_NORMAL);// Create a window for display.
	//cv::imshow(titile, dst_image);                   // Show our image inside it.
	cv::imwrite("e:/tmp/" + titile + ".jpg", dst_image);
	//cv::waitKey(0);                                          // Wait for a keystroke in the window
	
	return true;
}