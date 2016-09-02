#include "functions.h"

using namespace cv;
using namespace std;

#ifdef _DEBUG
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
#else
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#endif

int main()
{

	Mat image;
	image = imread("E:\\data\\pinball.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
	/*
	int row = 10;
	int col = 2;
	image.create(row, col, CV_8UC3);
	for (int i = 0; i < row; i++)
	{
		uchar *ptr = image.ptr<uchar>(i);
		for (int j = 0; j < col*image.channels(); j++)
		{
			ptr[j] = j % 255;
		}
	}
	*/
	Mat continuousRGBA(image.rows, image.cols, CV_8UC4);
	TIMED("convert to 4 channel")
	{
		
		cv::cvtColor(image, continuousRGBA, CV_RGB2RGBA, 4);
	}
	/*
	cv::namedWindow("test_RGBA", cv::WINDOW_NORMAL);// Create a window for display.
	cv::imshow("test_RGBA", continuousRGBA);                   // Show our image inside it.
	//cv::imwrite("e:/tmp/1.jpg", dst_image);
	cv::waitKey(0);                                          // Wait for a keystroke in the window
	*/
	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	/*
	for (int i = 0; i < row; i++)
	{
		uchar *ptr = image.ptr<uchar>(i);
		for (int j = 0; j < col*image.channels(); j++)
		{
			printf("i=%d,j=%d,c=%d,v=%u\n",i,j/3,j%3,ptr[j]);
		}
	}
	*/

	initCUDA();
	///to avoid cuda init lantancy
	uchar * head;
	checkCudaErrors(cudaMalloc((void**)&head, 10 * sizeof(uchar)));
	


	printf("\n\nnow start no pitch test\n");
	testGlobalReadNoPitch(image, "testGlobalReadNoPitch");

	printf("\n\nnow start pitch test\n");
	testGlobalReadPitch(image,"testGlobalReadPitch");

	printf("\n\nnow start tex no pitch test\n");
	testTexReadNoPitch(image,"testTexReadNoPitch");
	

	printf("\n\nnow start tex pitch test\n");
	testTexReadPitch(image, "testTexReadPitch");
	
	printf("\n\nnow start cuArray test\n");
	testcuArrayRead(image, "testcuArray");
	

	///////////////////////////////////////////////////////////////////4 channels
	printf("\n\nnow start no pitch 4 channels test\n");
	testGlobalReadNoPitch(continuousRGBA, "testGlobalReadNoPitch4c");

	printf("\n\nnow start pitch 4 channels test\n");
	testGlobalReadPitch(continuousRGBA, "testGlobalReadPitch4c");

	printf("\n\nnow start tex no pitch 4 channels test\n");
	testTexReadNoPitch(continuousRGBA, "testTexReadNoPitch4c");


	printf("\n\nnow start tex pitch 4 channels test\n");
	testTexReadPitch(continuousRGBA, "testTexReadPitch4c");

	printf("\n\nnow start cuArray 4 channels test\n");
	testcuArrayRead(continuousRGBA, "testcuArray4c");

	checkCudaErrors(cudaFree(head));
	cleanCUDA();

	cv::waitKey(0);
	return 1;
}