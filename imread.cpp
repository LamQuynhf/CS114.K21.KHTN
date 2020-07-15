#include<opencv2\opencv.hpp>

using namespace cv;

int main(){
	Mat img = imread("img-0242.png", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Sample", img);	
	waitKey(0);
	return 0;
}
