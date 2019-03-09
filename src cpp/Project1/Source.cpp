// 2018-01-03. 
// https://docs.opencv.org/3.4.0/d8/d01/tutorial_discrete_fourier_transform.html
// https://marketplace.visualstudio.com/items?itemName=VisualCPPTeam.ImageWatch2017#overview
// Discrete Fourier Transform
// 2018-01-06
// it gives the same result as matlab script
// 2018-01-08
// moving ROI works well
// 2019-03-09
// chenged variable names

//#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <conio.h>

using namespace cv;
using namespace std;

void help()
{
	cout << endl
		<< "This program demonstrated the use of the discrete Fourier transform (DFT). "	<< endl
		<< "The dft of an image is taken and it's power spectrum is displayed."				<< endl;
}

// Functions rearranges quadrants of Fourier image  so that the origin is at the image center
void fftshift(const Mat& inputImg, Mat& outputImg)
{
	// crop the spectrum, if it has an odd number of rows or columns
	outputImg = inputImg(Rect(0, 0, inputImg.cols & -2, inputImg.rows & -2));
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = outputImg.cols / 2;
	int cy = outputImg.rows / 2;
	Mat q0(outputImg, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(outputImg, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(outputImg, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(outputImg, Rect(cx, cy, cx, cy)); // Bottom-Right
	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

// Function calculates PSD(Power spectrum density) by fft with two flags
// flag = 0 means to return PSD
// flag = 1 means to return log(PSD)
void CalcPSD(const Mat& inputImg, Mat& outputImg, int flag = 0)
{
	Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);            // this way the result may fit in the source matrix
	split(complexI, planes);            // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

	float *p;
	p = planes[0].ptr<float>(0);
	p[0] = 0;								//	planes[0].at<float>(0) = 0;
	p = planes[1].ptr<float>(0);
	p[0] = 0;								//	planes[1].at<float>(0) = 0;

	// PSD calculation
	Mat imgPSD;
	magnitude(planes[0], planes[1], imgPSD);				//imgPSD = sqrt(Power spectrum density)
	pow(imgPSD, 2, imgPSD);									//it needs ^2 in order to get PSD
	outputImg = imgPSD;

	// switch PSD to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	Mat imglogPSD;
	if (flag)
	{
		imglogPSD = imgPSD + Scalar::all(1);						//switch to logarithmic scale
		log(imglogPSD, imglogPSD);									//imglogPSD = log(PSD)
		//Mat imgPhase;
		//phase(planes[0], planes[1], imgPhase);
		outputImg = imglogPSD;
	}
}

// Function calculates autocorrelation function
// InputPSD - PSD
// outputImg - ACF
void CalcACF(Mat& InputPSD, Mat& outputImg)
{
	float * p;
	Mat imgACF;
	Mat planesACF[2] = { Mat_<float>(InputPSD.clone()), Mat::zeros(InputPSD.size(), CV_32F) };
	merge(planesACF, 2, imgACF);     // Add to the expanded another plane with zeros
									 //dft(imgACF, imgACF, DFT_INVERSE + DFT_SCALE);
	idft(imgACF, imgACF);            // this way the result may fit in the source matrix
	split(imgACF, planesACF);
	p = planesACF[0].ptr<float>(0);
	p[0] = 0;
	outputImg = planesACF[0];
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	*(Point*)userdata = Point(x, y);
}

int main()
{
	help();
	Mat img = imread("D:\\home\\programming\\vc\\new\\2_OpenCV official tutorial\\4_Discrete Fourier Transform\\input\\00000110_resized.TIF");

	if (img.empty()) //check whether the image is loaded or not
	{
		cout << "ERROR : Image cannot be loaded..!!" << endl;
		//system("pause"); //wait for a key press
		return -1;
	}

	Mat imgGray;
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	namedWindow("Input image with Rect");
	//set the callback function for any mouse event
	Point p_origin;
	setMouseCallback("Input image with Rect", CallBackFunc, (void*)& p_origin);
	while (true)
	{
		cout << p_origin << endl;
		Rect box(p_origin, Size2i(128,128));
		if (box.br().x > imgGray.cols)
			box.x = imgGray.cols - box.size().width;
		if (box.br().y > imgGray.rows)
			box.y = imgGray.rows - box.size().height;
		Mat img_temp = imgGray.clone();
		Mat imgRoi = imgGray(box).clone();
		rectangle(img_temp, box.tl(), box.br(), Scalar(255));
		Mat imgPSD, imglogPSD, imgACF;
		CalcPSD(imgRoi, imgPSD);
		CalcPSD(imgRoi, imglogPSD, true);
		CalcACF(imgPSD, imgACF);
		sqrt(imgPSD, imgPSD);		// in order to improve visualisation
		sqrt(imgPSD, imgPSD);

		fftshift(imgPSD, imgPSD);
		fftshift(imglogPSD, imglogPSD);
		fftshift(imgACF, imgACF);

		normalize(imgRoi, imgRoi, 0, 255, NORM_MINMAX);
		normalize(imgPSD, imgPSD, 0, 1, NORM_MINMAX);
		normalize(imglogPSD, imglogPSD, 0, 1, NORM_MINMAX);
		normalize(imgACF, imgACF, 0, 1, NORM_MINMAX);

		imshow("Input image with Rect", img_temp);
		imshow("ROI", imgRoi);
		imshow("imgPSD", imgPSD);
		imshow("imglogPSD", imglogPSD);
		imshow("imgACF", imgACF);

		int iKey = waitKey(50);

		//if user press 'ESC' key
		if (iKey == 27)
			break;
	}
	return 0;
}