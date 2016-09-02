#include "functions.h"
///type,dim,mode,
texture<uchar, 1, cudaReadModeElementType> tex;

texture<uchar4, 1, cudaReadModeElementType> tex4;

__global__ void copyImageTexNoPitch(uchar * src, uchar * dst, uint imageH, uint imageW, uint channel)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < imageW) && (y < imageH))
	{
		if (channel == 3)
		{

			int index = channel * (y*imageW + x);
			//uchar* ptr_src = src + index;
			uchar* ptr_dst = dst + index;
			ptr_dst[0] = tex1Dfetch(tex, index + 0);
			ptr_dst[1] = tex1Dfetch(tex, index + 1);
			ptr_dst[2] = tex1Dfetch(tex, index + 2);
		}
		else ///channel ==4
		{
			int index = (y*imageW + x);
			uchar4* ptr_dst = (uchar4*)dst + index;
			ptr_dst[0] = tex1Dfetch(tex4, index + 0);
		}
	}
}

bool testTexReadNoPitch(cv::Mat &image, std::string  titile)
{

	uchar *devPtr_src, *devPtr_dst;
	size_t size = image.cols*image.rows*sizeof(uchar)* image.channels();
	cv::Mat dst_image;
	///we need a channel desc,we have 1,2,4 channel,no 3!
	cudaChannelFormatDesc cf;

	if (image.type() == CV_8UC3)
	{
		TIMED("malloc host dst memory")
		{
			dst_image.create(image.rows, image.cols, CV_8UC3);
		}
		cf = cudaCreateChannelDesc<uchar>();
	}
	else if (image.type() == CV_8UC4)
	{
		TIMED("malloc host dst memory")
		{
			dst_image.create(image.rows, image.cols, CV_8UC4);
		}
		cf = cudaCreateChannelDesc<uchar4>();
	}
	

	// specify mutable texture reference parameters
	tex.normalized = 0;
	///cudaFilterModeLinear is only supported by float!
	tex.filterMode = cudaFilterModePoint;
	tex.addressMode[0] = cudaAddressModeClamp;

	
	TIMED("malloc device dst memory")
	{
		checkCudaErrors(cudaMalloc((void**)&devPtr_src, size));
		checkCudaErrors(cudaMalloc((void**)&devPtr_dst, size));
	}
	

	

	
	if (image.type() == CV_8UC3)
	{
		// bind texture reference to array
		checkCudaErrors(cudaBindTexture(NULL, &tex, devPtr_src, &cf, size));
	}
	else if (image.type() == CV_8UC4)
	{
		// bind texture reference to array
		checkCudaErrors(cudaBindTexture(NULL, &tex4, devPtr_src, &cf, size));
	}



	TIMED("cudaMemcpyHostToDevice")
	{
		checkCudaErrors(cudaMemcpy(devPtr_src, image.data, size, cudaMemcpyHostToDevice));
	}

	int width = image.cols;
	int height = image.rows;
	int channel = image.channels();

	dim3 blockSize(16, 16, 1);
	dim3 gridSize(((uint)width + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);

	TIMED("copyImage")
	{
		for (int i = 0; i < 10000; i++)
		{
			copyImageTexNoPitch << <gridSize, blockSize >> >(devPtr_src, devPtr_dst, (uint)height, (uint)width, (uint)channel);
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



	//cv::namedWindow(titile, cv::WINDOW_NORMAL);// Create a window for display.
	//cv::imshow(titile, dst_image);                   // Show our image inside it.
	cv::imwrite("e:/tmp/" + titile + ".jpg", dst_image);
	//cv::waitKey(0);                                          // Wait for a keystroke in the window
	return true;
}