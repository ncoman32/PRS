// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <random>

//utility struct to hold a,b,c parameters of a line equation
// aX + bY + c = 0
struct lineEq {
	float a = 0.0f;
	float b = 0.0f;
	float c = 0.0f;
};

//utility struct for the hought transform local maxima storage and sorting
struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

//Utility function to draw a full line 

double Slope(int x0, int y0, int x1, int y1) {
	return (double)(y1 - y0) / (x1 - x0);
}

void fullLine(cv::Mat img, cv::Point a, cv::Point b, cv::Scalar color) {
	double slope = Slope(a.x, a.y, b.x, b.y);

	Point p(0, 0), q(img.cols, img.rows);

	p.y = -(a.x - p.x) * slope + a.y;
	q.y = -(b.x - q.x) * slope + b.y;

	line(img, p, q, color, 1, 8, 0);
}


void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

										   // the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
										  //VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

																		 //computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

/************************************************************* P R S *************************************************************************/

//utility function to give endpoints of a set of points as pair<> in order to draw a line
Point* computeEndPoints(std::pair<float, float>* points, int numberOfPoints, float theta0, float theta1) {
	Point* endPoints = new Point[2];
	Point *results = new Point[numberOfPoints];
	float xMax = -10000;
	float xMin = 10000;
	for (int i = 0; i < numberOfPoints; i++) {
		results[i].x = points[i].first;
		results[i].y = (results[i].x *theta1) + theta0;
		if (results[i].y > xMax) {
			xMax = results[i].y;
			endPoints[0].x = results[i].x;
			endPoints[0].y = results[i].y;
		}
		if (results[i].y < xMin) {
			xMin = results[i].y;
			endPoints[1].x = results[i].x;
			endPoints[1].y = results[i].y;
		}
	}
	return endPoints;
}
//utility function to give endpoints of a set of points as pair<> in polar coordinates
Point* computeEndPointsBetaP(std::pair<float, float>* points, int numberOfPoints, float beta, float p)
{
	Point* endPoints = new Point[2];
	int xMax = -10000, xMin = 100000;
	Point *resultsMethod2 = new Point[numberOfPoints];
	//construct line & extract endpoints
	for (int i = 0; i < numberOfPoints; i++) {
		resultsMethod2[i].x = points[i].first;
		resultsMethod2[i].y = (p - (resultsMethod2[i].x * cos(beta))) / sin(beta);
		if (resultsMethod2[i].y > xMax) {
			xMax = resultsMethod2[i].y;
			endPoints[0].x = resultsMethod2[i].x;
			endPoints[0].y = resultsMethod2[i].y;
		}
		if (resultsMethod2[i].y < xMin) {
			xMin = resultsMethod2[i].y;
			endPoints[1].x = resultsMethod2[i].x;
			endPoints[1].y = resultsMethod2[i].y;
		}
	}
	return endPoints;
}
//utility function to give endpoints of a set of Point objects

void linearRegression_Lab01() {
	//read from file
	std::fstream pointsFile;
	pointsFile.open("LinearRegressionPoints/points0.txt");
	int numberOfPoints = 0;
	pointsFile >> numberOfPoints;

	std::pair<float, float>  * points = new std::pair<float, float>[numberOfPoints];
	for (int i = 0; i < numberOfPoints; i++) {
		pointsFile >> points[i].first;
		pointsFile >> points[i].second;
	}

	int height = 250;
	int width = 250;
	Mat whiteImg1(height, width, CV_8UC3, CV_RGB(255, 255, 255));
	Mat whiteImg1b(height, width, CV_8UC3, CV_RGB(255, 255, 255));
	Mat whiteImg2(height, width, CV_8UC3, CV_RGB(255, 255, 255));
	Mat whiteImg3(height, width, CV_8UC3, CV_RGB(255, 255, 255));
	Mat whiteImg4(height, width, CV_8UC3, CV_RGB(255, 255, 255));
	//plot points to an image
	for (int i = 0; i < numberOfPoints; i++) {
		if (points[i].first < 0 || points[i].second < 0) {
			continue;
		}
		else {
			drawMarker(whiteImg1, cv::Point(points[i].first, points[i].second), cv::Scalar(0, 0, 0), MARKER_CROSS, 3, 1);
			drawMarker(whiteImg1b, cv::Point(points[i].first, points[i].second), cv::Scalar(0, 0, 0), MARKER_CROSS, 3, 1);
			drawMarker(whiteImg2, cv::Point(points[i].first, points[i].second), cv::Scalar(0, 0, 0), MARKER_CROSS, 3, 1);
			drawMarker(whiteImg3, cv::Point(points[i].first, points[i].second), cv::Scalar(0, 0, 0), MARKER_CROSS, 3, 1);
			drawMarker(whiteImg4, cv::Point(points[i].first, points[i].second), cv::Scalar(0, 0, 0), MARKER_CROSS, 3, 1);
		}
	}
	//model 1  
	// y = theta0 + theta1*x
	float theta0 = 0.0f, theta1 = 0.0f;
	float xySum = 0.0f, xSum = 0.0f, ySum = 0.0f, xxSum = 0.0f, yySum = 0.0f, yMinusXSum = 0.0f;
	for (int i = 0; i < numberOfPoints; i++) {
		xySum += points[i].first*points[i].second;
		xSum += points[i].first;
		ySum += points[i].second;
		xxSum += points[i].first * points[i].first;
		yySum += points[i].second * points[i].second;
		yMinusXSum += points[i].second * points[i].second - points[i].first * points[i].first;
	}

	theta1 = ((numberOfPoints * xySum) - (xSum * ySum)) / ((numberOfPoints * xxSum) - (xSum * xSum));
	theta0 = 1.0f / numberOfPoints * (ySum - theta1*xSum);

	//computing the line and finding its endpoints
	Point* methodOneEndPoints = computeEndPoints(points, numberOfPoints, theta0, theta1);
	fullLine(whiteImg1, methodOneEndPoints[0], methodOneEndPoints[1], Scalar(0, 0, 255));
	imshow("MethodOne", whiteImg1);

	//model one Gradient Descent
	//y = theta0 + theta1*x
	//cost_funtion = J(theta) = 1/2 * sum(f(xi) - yi)^2
	//cost function is splitted in two, J(theta0), J(theta1)
	//J'(theta0) = sum(f(xi)-yi)
	//J'(theta1) = sum( (f(xi)-yi)xi) )  . Gradient = (J'(theta0), J'(theta1) ) 
	// use the rule : theta_new = theta_old + learning_rate[alpha] *gradient
	//use an error 
	theta0 = 0.5;
	theta1 = 0.5;
	float alfa = 0.000001f, updateTheta0 = 0.0f, updateTheta1 = 0.0f;
	float updateError = 0.001f;
	Point* gradientLinePoints = (Point*)malloc(2 * sizeof(Point));
	for (int k = 0; k < 100; k++) {
		float Jtheta0 = 0.0f, Jtheta1 = 0.0f;
		for (int i = 0; i < numberOfPoints; i++) {
			Jtheta0 += (theta0 + (theta1 *points[i].first)) - points[i].second;
			Jtheta1 += ((theta0 + (theta1 *points[i].first)) - points[i].second)*points[i].first;
		}
		std::cout << "Value of const function at step " << k
			<< ": J(theta) =[" << Jtheta0 << ", " << Jtheta1 << "]\n";

		updateTheta0 = theta0 - alfa * Jtheta0;
		updateTheta1 = theta1 - alfa * Jtheta1;
		if ((abs(theta0 - updateTheta0) < updateError) && (abs(theta1 - updateTheta1) < updateError)) {
			gradientLinePoints = computeEndPoints(points, numberOfPoints, theta0, theta1);
			break;
		}
		else {
			theta0 = updateTheta0;
			theta1 = updateTheta1;
		}

		std::cout << "Value of theta at step " << k
			<< ": [theta] =[" << theta0 << ", " << theta1 << "]\n";


		if (k % 10 == 0) {
			gradientLinePoints = computeEndPoints(points, numberOfPoints, theta0, theta1);
			line(whiteImg3, gradientLinePoints[0], gradientLinePoints[1], Scalar(0, 255, 0));
			imshow("1 - Gradient Intermediate", whiteImg3);
			waitKey(0);
		}
	}
	fullLine(whiteImg3, gradientLinePoints[0], gradientLinePoints[1], Scalar(0, 0, 255));
	imshow("Method One final result", whiteImg3);

	//model one closed version
	Mat A = Mat(numberOfPoints, 2, CV_32FC1);
	Mat b = Mat(numberOfPoints, 1, CV_32FC1);

	for (int c = 0; c < numberOfPoints; c++) {
		A.at<float>(c, 0) = points[c].first;
		A.at<float>(c, 1) = 1.0f;
		b.at<float>(c, 0) = points[c].second;
	}

	Mat thetaOpt = (A.t() * A).inv()*A.t()*b;
	//y= theta0 + xtheta1
	Point2f one(A.at<float>(10, 0), thetaOpt.at<float>(1, 0) + thetaOpt.at<float>(0, 0)* A.at<float>(10, 0));
	Point2f two(A.at<float>(30, 0), thetaOpt.at<float>(1, 0) + thetaOpt.at<float>(0, 0)* A.at<float>(20, 0));

	fullLine(whiteImg1b, one, two, Scalar(255, 0, 0));
	imshow("closed form", whiteImg1b);
	waitKey(0);

	//model 2 𝑥𝑐𝑜𝑠(𝛽) + 𝑦𝑠𝑖𝑛(𝛽) = p , [𝑐𝑜𝑠(𝛽) + 𝑠𝑖𝑛(𝛽)] =  normala, la distanta p de origine
	float beta = (-1.0f / 2.0f) * atan2(2.0f * xySum - ((2.0f / numberOfPoints) * xSum * ySum),
		yMinusXSum + (1.0f / numberOfPoints) * xSum * xSum - (1.0f / numberOfPoints) * ySum * ySum);

	float p = (1.0f / numberOfPoints) * (cos(beta) * xSum + sin(beta) * ySum);
	Point* methodTwoEndPoints = computeEndPointsBetaP(points, numberOfPoints, beta, p);
	fullLine(whiteImg2, methodTwoEndPoints[0], methodTwoEndPoints[1], Scalar(255, 0, 0));
	imshow("MethodTwo", whiteImg2);
	//model two Gradient Descent
	//xcos(beta) + ysin(beta) = p
	//cost_funtion = J(beta,p) =1/2 sum(xcos(beta) + ysin(beta) -p)^2
	//cost function is splitted in two, J(theta0), J(theta1)
	//J'(beta) = sum(xcos(beta) + ysin(beta) -p)(-xsin(beta) +ycos(beta))
	//J'(p) = -sum( xcos(beta) + ysin(beta) -p ) 
	// use the rule : theta_new = theta_old + learning_rate[alpha] *gradient
	//use an error 
	beta = 0.8;
	p = 0.3;
	alfa = 0.00005;
	updateError = 5.0f;
	float updateBeta = 0.0f, updateP = 0.0f;
	for (int k = 0; k < 200; k++) {
		float jBeta = 0.0f, jP = 0.0f;
		for (int i = 0; i < numberOfPoints; i++) {
			jBeta += (points[i].first * cos(beta) + points[i].second * sin(beta) - p)
				*(-points[i].first * sin(beta) + points[i].second * cos(beta));
			jP -= (points[i].first * cos(beta) + points[i].second * sin(beta) - p);
		}
		std::cout << "Value of const function at step " << k
			<< ": J(beta,p) =[" << jBeta << ", " << jP << "]\n";

		updateBeta = beta - alfa * jBeta;
		updateP = p - alfa * jP;
		if ((abs(beta - updateBeta) < updateError) && (abs(p - updateP) < updateError)) {
			gradientLinePoints = computeEndPointsBetaP(points, numberOfPoints, beta, p);
			break;
		}
		else {
			beta = updateBeta;
			p = updateP;
		}

		std::cout << "Value of theta at step " << k
			<< ": [beta,p] =[" << beta << ", " << p << "]\n";
		if (k % 10 == 0) {
			gradientLinePoints = computeEndPointsBetaP(points, numberOfPoints, beta, p);
			line(whiteImg4, gradientLinePoints[0], gradientLinePoints[1], Scalar(0, 255, 0));
			imshow("Method Two Gradient Intermediate", whiteImg4);
			waitKey(0);
		}
	}
	fullLine(whiteImg4, gradientLinePoints[0], gradientLinePoints[1], Scalar(0, 0, 255));
	imshow("Method Two final result", whiteImg4);
	//to do - model 3

	waitKey(0);
}

double distancePointToLine(Point one, lineEq line) {
	return fabs(one.x * line.a + one.y*line.b + line.c) / sqrt((line.a*line.a) + (line.b*line.b));
}

 
lineEq lineEquation(Point one, Point two) {
	lineEq line;
	line.a = one.y - two.y;
	line.b = two.x - one.x;
	line.c = one.x*two.y - two.x*one.y;
	return line;
}

void RANSAC_lab02() {
	srand(time(NULL));
	Mat img = imread("images_ransac/points5.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int height = img.rows;
	int width = img.cols;
	std::vector<Point> points;
	int numberOfPoints = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == 0) {
				Point newPoint(j, i);
				points.push_back(newPoint);
				numberOfPoints++;
			}
		}
	}
	//ransaca variables
	const float t = 10.0f;
	const float p = 0.99f;
	const float q = 0.8f;
	const float s = 2.0f;
	const float N = log(1.0f - p) / log(1.0f - pow(q, s));
	const int  T = q*numberOfPoints;


	//line which keeps track of best line found in case all iterations are made
	lineEq bestline;
	int numberOfInLiers = 0;
	int maxInLiersCount = 0;
	lineEq line;
	boolean drewLine = false;
	int iteration;
	for (iteration = 0; iteration < N; iteration++) {
		//re initialize number of in liers for each iteration
		numberOfInLiers = 0;

		int pointIndexOne = rand() % numberOfPoints;
		int pointIndexTwo = rand() % numberOfPoints;
		while (pointIndexOne == pointIndexTwo) {
			pointIndexTwo = rand() % numberOfPoints;
		}
		line = lineEquation(points.at(pointIndexOne), points.at(pointIndexTwo));

		for (int i = 0; i < numberOfPoints; i++) {
			double distance = distancePointToLine(points.at(i), line);
			if (distance <= t) {
				numberOfInLiers++;
			}
		}
		if (iteration == 0) {
			maxInLiersCount = numberOfInLiers;
		}
		std::cout << "Iteration " << iteration << ": " << pointIndexOne << " <-indexes->" << pointIndexTwo;
		std::cout << " Inlier: " << numberOfInLiers << " Threshold: " << T << "\n";
		//if number of inliers big enough draw line
		if (numberOfInLiers > T) {
			Point2d one(0.0, -line.c / line.b);
			Point2d two(width, (-line.c - line.a*width) / line.b);

			Point2f oneY(-bestline.c / bestline.a, 0.0f);
			Point2f twoY((-bestline.c - bestline.b*height) / bestline.a, height);
			//cv::line(img, oneY, twoY, Scalar(0, 0, 255), 2);

			cv::line(img, one, two, Scalar(0, 0, 255));
			imshow("inliers>T", img);
			waitKey(0);
			drewLine = true;
			break;
		}
		else {
			if (numberOfInLiers > maxInLiersCount) {
				maxInLiersCount = numberOfInLiers;
				bestline = line;
			}
		}

	}
	if (drewLine == false) {

		Point2f one(0.0f, -bestline.c / bestline.b);
		Point2f two(img.cols, (-bestline.c - (bestline.a*width)) / bestline.b);

		Point2f oneY(-bestline.c / bestline.a, 0.0f);
		Point2f twoY((-bestline.c - bestline.b*height) / bestline.a, height);

		cv::line(img, one, two, Scalar(0, 0, 255), 1);
		cv::line(img, oneY, twoY, Scalar(0, 0, 255), 2);

		imshow("Best Inliers Ser", img);
		std::cout << "Best consensus set had: " << maxInLiersCount << " Inliers";
		waitKey(0);

	}

}
 

double degToRad(int deg) {

	return (deg*PI) / 180.0;
}
 
void HoughTransform_lab03() {

	Mat srcImg = imread("images_hough/edge_simple.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat srcImgRgb(srcImg.size(), CV_8UC3);
	cvtColor(srcImg, srcImgRgb, CV_GRAY2RGB);
	//compute maxP  using pythagorean theorem . maxP =sqrt( widht^2 + height^2)
	int maxP = sqrt(srcImg.rows*srcImg.rows + srcImg.cols*srcImg.cols);
	int maxTheta = 360;
	int deltaTheta = 1;
	int deltaP = 1;
	int p;
	Mat hough = Mat::zeros(360, maxP + 1, CV_32SC1);
	Mat houghImg;
	//detect edges with canny edge 
	Mat gauss;
	Mat imageWithEdges;
	double k = 0.4;
	int pH = 50;
	int pL = (int)k*pH;
	GaussianBlur(srcImg, gauss, Size(5, 5), 0.8, 0.8);
	Canny(gauss, imageWithEdges, pL, pH, 3);


	for (int i = 0; i < imageWithEdges.rows; i++) {
		for (int j = 0; j < imageWithEdges.cols; j++) {
			//if edge point
			if (imageWithEdges.at<uchar>(i, j) == 255) {
				for (int theta = 0; theta < maxTheta; theta++) {
					p = j * cos(degToRad(theta)) + i * sin(degToRad(theta));
					if (p>0 && p < maxP) {
						hough.at<int>(theta, p)++;
					}
				}
			}
		}
	}
	double min, max;
	minMaxLoc(hough, &min, &max);
	hough.convertTo(houghImg, CV_8UC1, 255.f / (float)max);
	//peak struct array to store the found peaks
	peak peaks[20000];
	int numberOfPeaks = 0;
	bool isGreater;
	//use 9x9 mask
	for (int theta = 4; theta < maxTheta - 4; theta += deltaTheta) {
		for (int p = 4; p < maxP - 4; p += deltaP) {
			isGreater = true;
			for (int k = theta - 4; k <= theta + 4; k++) {
				//test current element against all elements in the k x k window
				for (int l = p - 4; l <= p + 4; l++) {
					if (hough.at<int>(theta, p) == 0 || hough.at<int>(theta, p) < hough.at<int>(k, l)) {
						isGreater = false;
					}
				}
			}
			if (isGreater == true) {
				peaks[numberOfPeaks].hval = hough.at<int>(theta, p);
				peaks[numberOfPeaks].ro = p;
				peaks[numberOfPeaks].theta = theta;
				numberOfPeaks++;
			}
		}
	}
	std::sort(peaks, peaks + numberOfPeaks);
	//convert polar coordinates in cartesian coordinates and draw lines
	for (int i = 0; i < 10; i++) {
		Point pt1, pt2;
		double a = cos(degToRad(peaks[i].theta)), b = sin(degToRad(peaks[i].theta));
		double x0 = a*peaks[i].ro, y0 = b*peaks[i].ro;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(srcImgRgb, pt1, pt2, Scalar(0, 255, 0));
		std::cout << peaks[i].hval << std::endl;
	}
	imshow("houghImg", houghImg);
	imshow("result", srcImgRgb);
	imshow("canny", imageWithEdges);
	waitKey(0);
}

	//utility minimum funct
	int getMin(int a, int b, int c, int d) {
		return min(a, min(b, min(c, d)));
	}


	//Chamfer DT, 3x3 mask splitted in 2
	//Chamfer DT, 3x3 mask splitted in 2
	Mat distanceTransform(Mat srcImg) {
		Mat DT = srcImg.clone();
		int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
		int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
		int weights[8] = { 3,2,3, 2, 2, 3, 2, 3 };
		int min = 0, currentValue = 0;
		//first pass
		for (int i = 1; i < srcImg.rows - 1; i++) {
			for (int j = 1; j < srcImg.cols - 1; j++) {
				min = INT_MAX;
				//use first part of the mask
				for (int k = 0; k < 4; k++) {
					currentValue = DT.at<uchar>(i + di[k], j + dj[k]) + weights[k];
					if (currentValue < min) {
						min = currentValue;
					}
				}
				if (min < DT.at<uchar>(i, j))
					DT.at<uchar>(i, j) = min;
			}
		}
		//second pass
		for (int i = srcImg.rows - 2; i > 0; i--) {
			for (int j = srcImg.cols - 2; j > 0; j--) {
				min = INT_MAX;
				for (int k = 4; k < 8; k++) {
					currentValue = DT.at<uchar>(i + di[k], j + dj[k]) + weights[k];
					if (currentValue < min) {
						min = currentValue;
					}
				}
				if (min < DT.at<uchar>(i, j))
					DT.at<uchar>(i, j) = min;
			}

		}
		return DT;
	}

Point getMassCenterForContour(Mat srcImg) {
	//find the mass center for template image
	double area = 0.0;
	double mr = 0.0, mc = 0.0;
	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if (srcImg.at<uchar>(i, j) == 0) {
				area++;
				mr += j;
				mc += i;
			}
		}
	}
	mr /= area;
	mc /= area;
	Point massCenter(mc, mr);
	return massCenter;

}
Mat translateImage(Mat srcImg, Point displacement) {
	Mat translatedImg = Mat(srcImg.rows, srcImg.cols, CV_8UC1);
	for (int i = abs(displacement.y); i < srcImg.rows - abs(displacement.y); i++) {
		for (int j = abs(displacement.x); j < srcImg.cols - abs(displacement.x); j++) {
			if (srcImg.at<uchar>(i, j) == 0) {
				translatedImg.at<uchar>(i - displacement.y, j - displacement.x) = 0;
			}
			else {
				translatedImg.at<uchar>(i - displacement.y, j - displacement.x) = 255;
			}
		}
	}
	return translatedImg;
}

/*The less the pattern matching score is the more similar is the
unknown object to the template*/
void patternMatchingWithDT_lab04() {

	Mat templateImage = imread("images_DT_PM/PatternMatching/template.bmp", IMREAD_GRAYSCALE);
	Mat DT = distanceTransform(templateImage);
	Mat unkownObject1 = imread("images_DT_PM/PatternMatching/unknown_object1.bmp", IMREAD_GRAYSCALE);
	Mat unkownObject2 = imread("images_DT_PM/PatternMatching/unknown_object2.bmp", IMREAD_GRAYSCALE);

	//mass centers
	Point templateMassCenter = getMassCenterForContour(templateImage);
	Point objectOneMassCenter = getMassCenterForContour(unkownObject1);
	Point objectTwoMassCenter = getMassCenterForContour(unkownObject2);

	//translate first image
	Point displacement(objectOneMassCenter.x - templateMassCenter.x, objectOneMassCenter.y - templateMassCenter.y);
	Mat translatedImg = translateImage(unkownObject1, displacement);


	//perform score for object 1
	double contourPixelsCount = 0.0;
	double scoreOne = 0.0;
	for (int i = 0; i < translatedImg.rows; i++) {
		for (int j = 0; j < translatedImg.cols; j++) {
			if (translatedImg.at<uchar>(i, j) == 0) {
				scoreOne += (double)DT.at<uchar>(i, j);
				contourPixelsCount += 1.0;
			}
		}
	}
	scoreOne /= contourPixelsCount;

	displacement.x = objectTwoMassCenter.x - templateMassCenter.x;
	displacement.y = objectTwoMassCenter.y - templateMassCenter.y;
	translatedImg = translateImage(unkownObject2, displacement);
	contourPixelsCount = 0.0;
	double scoreTwo = 0.0;
	for (int i = 0; i < translatedImg.rows; i++) {
		for (int j = 0; j < translatedImg.cols; j++) {
			if (translatedImg.at<uchar>(i, j) == 0) {
				scoreTwo += (double)DT.at<uchar>(i, j);
				contourPixelsCount += 1.0;
			}
		}
	}
	scoreTwo /= contourPixelsCount;

	std::cout << "score for object1(pedestrian): " << scoreOne << std::endl;
	std::cout << "score for object2(leaf): " << scoreTwo << std::endl;
	imshow("template", templateImage);
	imshow("object1", unkownObject1);
	imshow("object2", unkownObject2);
	waitKey(0);

}

void statisticalAnalysis_lab05() {
	const int p = 400;
	//each image has a dimension of 19x19
	const int N = 19 * 19;
	//each entry in the intensity matrix is a sample of a featre
	//a feature is a 19*19 random variable vector

	//intensity matrix
	Mat I = Mat(p, N, CV_8UC1);
	char folder[256] = "faces";
	char fname[256];
	for (int i = 1; i <= 400; i++) {
		sprintf(fname, "%s/face%05d.bmp", folder, i);
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0;
		for (int r = 0; r < img.rows; r++) {
			for (int c = 0; c < img.cols; c++) {
				I.at<uchar>(i - 1, k) = img.at<uchar>(r, c);
				k++;
			}
		}
	}
	//compute mean for each feature
	std::vector<double> meanValues;
	for (int c = 0; c < N; c++) {
		double mean = 0.0;
		for (int r = 0; r < p; r++) {
			mean += (double)I.at<uchar>(r, c);
		}
		mean = mean / p;
		meanValues.push_back(mean);
	}
	//write means  to file 
	std::ofstream meanValuesFile;
	meanValuesFile.open("meanValues.txt");
	for (int i = 0; i < N; i++) {
		meanValuesFile << meanValues[i] << std::endl;
	}

	//compute standard deviation for each feature
	std::vector<double> stdDevValues;
	for (int c = 0; c < N; c++) {
		double stdDev = 0.0;
		for (int r = 0; r < p; r++) {
			stdDev += ((double)I.at<uchar>(r, c) - meanValues.at(c))
				* ((double)I.at<uchar>(r, c) - meanValues.at(c));
		}
		stdDev = stdDev / (double)p;
		stdDevValues.push_back(sqrt(stdDev));
	}

	//compute C
	Mat C = Mat(N, N, CV_64FC1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			double c = 0.0;
			for (int k = 0; k < p; k++) {
				c += ((double)I.at<uchar>(k, i) - meanValues.at(i))
					* ((double)I.at<uchar>(k, j) - meanValues.at(j));
			}
			c = c / (double)p;
			C.at<double>(i, j) = c;
		}
	}
	//write to C to file
	std::ofstream covMatFile;
	covMatFile.open("covMat.txt");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (j == N - 1) {
				covMatFile << C.at<double>(i, j);
				covMatFile << "/n";
			}
			else {
				covMatFile << C.at<double>(i, j) << ", ";
			}

		}
	}
	Mat corMat = Mat(N, N, CV_64FC1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j) {
				corMat.at<double>(i, j) == 1.0;
			}
			else {
				corMat.at<double>(i, j) = C.at<double>(i, j) / (stdDevValues.at(i) * stdDevValues.at(j));
			}
		}
	}

	Mat corChart = Mat(256, 256, CV_8UC1);
	//(5,4), (5,14)
	for (int i = 0; i < p; i++) {
		corChart.at<uchar>(I.at<uchar>(i, 5 * 19 + 4), I.at<uchar>(i, 5 * 19 + 14)) = 0;
	}
	imshow("corChart", corChart);
	std::cout << corMat.at<double>(5 * 19 + 4, 5 * 19 + 14);
	waitKey(0);
}


int getMinimumElementIndex(float* vector, int n) {

	float minimum = vector[0];
	int index = 0;
	for (int i = 0; i < n; i++) {
		if (vector[i] < minimum) {
			index = i;
			minimum = vector[i];
		}
	}
	return index;
}

struct punct {
	Point p;
	int color;
};

void KMeansClustering_labo06_setOfPoints(const int K) {
	//obtain rgb image from original image
	Mat srcImg = imread("Images_Kmeans/points2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat srcImgRgb(srcImg.size(), CV_8UC3);
	cvtColor(srcImg, srcImgRgb, CV_GRAY2RGB);

	const int height = srcImgRgb.rows;
	const int width = srcImgRgb.cols;
	//generate random colors
	std::default_random_engine gen;
	gen.seed(time(NULL));
	std::uniform_int_distribution<int> dist_img(0, 255);
	Vec3b *colors = (Vec3b*)malloc(K * sizeof(Vec3b));

	for (int i = 0; i < K; i++) {
		colors[i] = { (uchar)dist_img(gen), (uchar)dist_img(gen), (uchar)dist_img(gen) };
	}
	//initialize  k centers 
	std::vector<Point> points;
	int pointsCounter = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (srcImg.at<uchar>(i, j) == 0) {
				Point newPoint(i, j);
				points.push_back(newPoint);
				pointsCounter++;
			}
		}
	}

	Point *m = (Point*)malloc(K * sizeof(Point));
	Point *m_new = (Point*)calloc(K, sizeof(Point));
	std::uniform_int_distribution<int> points_distribution(0, pointsCounter - 1);
	for (int i = 0; i < K; i++) {
		m[i] = points.at((int)points_distribution(gen));
		std::cout << m[i].x << " " << m[i].y << "\n";
	}



	float* distances = (float*)malloc(K * sizeof(float));
	punct* puncte = (punct*)malloc(pointsCounter * sizeof(punct));
	bool changed = true;

	while (changed) {
		//Assignment
		for (int i = 0; i < pointsCounter; i++) {
			for (int k = 0; k < K; k++) {
				distances[k] = sqrt(((m[k].x - points[i].x) * (m[k].x - points[i].x)) + ((m[k].y - points[i].y) * (m[k].y - points[i].y)));
			}
			puncte[i].color = getMinimumElementIndex(distances, K);
			puncte[i].p = points.at(i);
		}

		//vectors used to update centers
		int*sumX = (int*)malloc(K * sizeof(int));
		int*sumY = (int*)malloc(K * sizeof(int));
		int*numerOfPoints = (int*)malloc(K * sizeof(int));
		for (int c = 0; c < K; c++) {
			sumX[c] = 0;
			sumY[c] = 0;
			numerOfPoints[c] = 0;
		}

		//Update
		for (int i = 0; i < pointsCounter; i++) {
			sumX[puncte[i].color] += puncte[i].p.x;
			sumY[puncte[i].color] += puncte[i].p.y;
			numerOfPoints[puncte[i].color]++;
		}
		for (int c = 0; c < K; c++) {
			m[c].x = sumX[c] / numerOfPoints[c];
			m[c].y = sumY[c] / numerOfPoints[c];
		}
		//check if assignment function produced new centers
		changed = false;
		for (int v = 0; v < K; v++) {
			if (m[v].x != m_new[v].x || m[v].y != m_new[v].y) {
				changed = true;
			}
		}
		for (int cnt = 0; cnt < K; cnt++) {
			m_new[cnt] = m[cnt];
		}
	}
	Mat voronoi = srcImgRgb.clone();
	for (int i = 0; i < pointsCounter; i++) {
		srcImgRgb.at<Vec3b>(puncte[i].p.x, puncte[i].p.y) = colors[puncte[i].color];
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < K; k++) {
				distances[k] = sqrt(((m[k].x - i) * (m[k].x - i) + ((m[k].y - j) * (m[k].y - j))));
			}
			voronoi.at<Vec3b>(i, j) = colors[getMinimumElementIndex(distances, K)];
		}
	}

	imshow("result", srcImgRgb);
	imshow("voroni_mozaicare", voronoi);
	waitKey(0);
}

void KMeansClustering_lab06_grayScale(int noOfIterations) {
	Mat srcImg = imread("images_Kmeans/img01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//initialize  k centers 
	const int height = srcImg.rows;
	const int width = srcImg.cols;
	const int K = 10;
	Mat rez = Mat(height, width, CV_8UC1);
	uchar m[K]; // <--- the centers
	std::default_random_engine gen;
	std::uniform_int_distribution<int> dist_img(0, 255);
	for (int i = 0; i < K; i++) {
		m[i] = dist_img(gen);
	}
	Mat L = Mat(height, width, CV_8UC1);
	Mat newL = Mat(height, width, CV_8UC1);
	float distances[K];
	for (int count = 0; count < noOfIterations; count++) {
		//Assignment function
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int k = 0; k < K; k++) {
					distances[k] = sqrt((m[k] - srcImg.at<uchar>(i, j)) *  (m[k] - srcImg.at<uchar>(i, j)));
				}
				L.at<uchar>(i, j) = getMinimumElementIndex(distances, K);
			}
		}
		if (count > 0) {
			//check if assignment function produced new results
			cv::Mat diff = L != newL;
			bool eq = cv::countNonZero(diff) == 0;
			if (eq == true) {
				std::cout << "assignent function produced same results at iteration: " << count;
				break;
			}
		}
		//update centers
		int sumsGray[K];
		int noOfPoints[K];
		for (int i = 0; i < K; i++) {
			sumsGray[i] = 0;
			noOfPoints[i] = 0;
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				sumsGray[L.at<uchar>(i, j)] += srcImg.at<uchar>(i, j);
				noOfPoints[L.at<uchar>(i, j)]++;
			}
		}
		for (int c = 0; c < K; c++) {
			m[c] = sumsGray[c] / noOfPoints[c];
		}
		L.copyTo(newL);
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			rez.at<uchar>(i, j) = m[L.at<uchar>(i, j)];
		}
	}
	imshow("gray", rez);
	waitKey(0);

}

void KMeansClustering_lab06_RGB(int noOfIterations) {
	Mat srcImg = imread("images_Kmeans/img02.jpg", CV_LOAD_IMAGE_COLOR);
	const int height = srcImg.rows;
	const int width = srcImg.cols;
	Mat rez = Mat(height, width, CV_8UC3);
	//generate centers
	std::default_random_engine gen;
	std::uniform_int_distribution<int> dist_img(0, 255);
	const int K = 10;
	Vec3b m[K];
	for (int i = 0; i < K; i++) {
		m[i][0] = dist_img(gen);
		m[i][1] = dist_img(gen);
		m[i][2] = dist_img(gen);
	}

	Mat L = Mat(height, width, CV_8UC1);
	Mat newL = Mat(height, width, CV_8UC1);
	float distances[K];
	for (int count = 0; count < noOfIterations; count++) {
		//Assignment function
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				for (int k = 0; k < K; k++) {
					distances[k] = sqrt(
						((m[k][0] - srcImg.at<Vec3b>(i, j)[0]) * (m[k][0] - srcImg.at<Vec3b>(i, j)[0])) +
						((m[k][1] - srcImg.at<Vec3b>(i, j)[1]) * (m[k][1] - srcImg.at<Vec3b>(i, j)[1])) +
						((m[k][2] - srcImg.at<Vec3b>(i, j)[2]) * (m[k][2] - srcImg.at<Vec3b>(i, j)[2]))
					);
				}
				L.at<uchar>(i, j) = getMinimumElementIndex(distances, K);
			}
		}
		if (count > 0) {
			//check if assignment function produced new results
			bool isEqual = (sum(L != newL) == Scalar(0, 0, 0));
			if (isEqual == true) {
				std::cout << "assignent function produced same results at iteration: " << count;
				break;
			}

		}
		//update centers
		int sumsR[K];
		int sumsG[K];
		int sumsB[K];
		int noOfPoints[K];
		for (int i = 0; i < K; i++) {
			sumsR[i] = 0;
			sumsG[i] = 0;
			sumsB[i] = 0;
			noOfPoints[i] = 0;
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				sumsB[L.at<uchar>(i, j)] += srcImg.at<Vec3b>(i, j)[0];
				sumsG[L.at<uchar>(i, j)] += srcImg.at<Vec3b>(i, j)[1];
				sumsR[L.at<uchar>(i, j)] += srcImg.at<Vec3b>(i, j)[2];
				noOfPoints[L.at<uchar>(i, j)]++;
			}
		}
		for (int c = 0; c < K; c++) {
			if (noOfPoints[c] > 0) {
				m[c][0] = sumsB[c] / noOfPoints[c];
				m[c][1] = sumsG[c] / noOfPoints[c];
				m[c][2] = sumsR[c] / noOfPoints[c];
			}
		}
		L.copyTo(newL);
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			rez.at<Vec3b>(i, j) = m[L.at<uchar>(i, j)];
		}
	}
	imshow("color", rez);
	waitKey(0);
}
/***********************************BAD COMPUTATIONS PROBABLY****************************************************/

void principalComponentAnalysis_lab07() {

	std::fstream pointsFile;
	pointsFile.open("PCA/pca2d.txt");
	//the number of points
	int numberOfPoints = 0;
	pointsFile >> numberOfPoints;
	//number of values in each feature
	int dimensionality = 0;
	pointsFile >> dimensionality;

	Mat points = Mat(numberOfPoints, dimensionality, CV_64FC1);

	for (int i = 0; i < numberOfPoints; i++) {
		for (int j = 0; j < dimensionality; j++) {
			pointsFile >> points.at<double>(i, j);
		}
	}
	pointsFile.close();
	//should be equal to the number of points
	int height = points.rows;
	//should be equal to the points dimensionality
	int width = points.cols;

	//the mean vector
	Mat miu = Mat(numberOfPoints, 1, CV_64FC1);
	for (int i = 0; i < height; i++) {
		double currentMiu = 0.0;
		for (int j = 0; j < width; j++) {
			currentMiu += points.at<double>(i, j);
		}
		miu.at<double>(i, 0) = (currentMiu / (double)dimensionality);
	}
	//one-row matrix containing only ones
	Mat onesRowMatrix = Mat(1, numberOfPoints, CV_64FC1);
	for (int i = 0; i < numberOfPoints; i++) {
		onesRowMatrix.at<double>(0, i) = 1.0;
	}
	//extract mean from the points set
	for (int i = 0; i < numberOfPoints; i++) {
		for (int j = 0; j < dimensionality; j++) {
			points.at<double>(i, j) -= miu.at<double>(i, 0);
		}
	}

	//compute covariance matrix as matrix multiplication
	Mat C = Mat(numberOfPoints, numberOfPoints, CV_64FC1);
	C = points.t()*points / (numberOfPoints - 1);

	//perform eigenvalues decomposition
	Mat Lambda, Q;
	eigen(C, Lambda, Q);
	Q = Q.t();
	for (int i = 0; i < Lambda.rows; i++) {
		for (int j = 0; j < Lambda.cols; j++) {
			std::cout << Lambda.at<double>(i, j) << "\t";
		}
		std::cout << std::endl;
	}

	//compute PCA coefficients 
	Mat XCoef = points * Q;


	waitKey(0);
	getchar();
	getchar();
}
/****************************************************************************************************************/


/***************************************************KNN Classifier***********************************************/
/* Blue(0-255) Green(256-511) Red(512 -767)*/
int* colorHistogram(Mat srcImg, int nrBins) {
	//convert to hsv before histogram computation
	Mat img_hsv;
	cvtColor(srcImg, img_hsv, CV_BGR2HSV);
	int* histogram = new int[nrBins * 3];
	const int binSize = 256 / nrBins;
	for (int c = 0; c < nrBins * 3; c++) {
		histogram[c] = 0;
	}
	if (nrBins == 256) {
		for (int i = 0; i < img_hsv.rows; i++) {
			for (int j = 0; j < img_hsv.cols; j++) {
				histogram[img_hsv.at<Vec3b>(i, j)[0]]++;
				histogram[nrBins + img_hsv.at<Vec3b>(i, j)[1]]++;
				histogram[(nrBins * 2) + img_hsv.at<Vec3b>(i, j)[2]]++;
			}
		}
	}
	else {
		int currentBin = 0;
		for (int i = 0; i < img_hsv.rows; i++) {
			for (int j = 0; j < img_hsv.cols; j++) {
				currentBin = 0;
				while (currentBin < nrBins) {
					uchar red = img_hsv.at<Vec3b>(i, j)[2];
					uchar green = img_hsv.at<Vec3b>(i, j)[1];
					uchar blue = img_hsv.at<Vec3b>(i, j)[0];
					if (blue >= currentBin*binSize && blue < (currentBin + 1)*binSize) {
						histogram[currentBin]++;
					}
					if (green >= currentBin*binSize && green < (currentBin + 1)*binSize) {
						histogram[nrBins + currentBin]++;
					}
					if (red >= currentBin*binSize && red < (currentBin + 1)*binSize) {
						histogram[(nrBins * 2) + currentBin]++;
					}
					currentBin++;
				}

			}
		}
	}
	return histogram;
}

//C = Mat of Floats
double getAccuracyFromConfusionMatrix(Mat C)
{
	double accuracy = 0.0;
	double nominator = 0.0;
	double denominator = 0.0;
	for (int i = 0; i < C.rows; i++) {
		for (int j = 0; j < C.cols; j++) {
			if (i == j) {
				nominator += C.at<float>(i, j);
			}
			denominator += C.at<float>(i, j);
		}

	}
	accuracy = nominator / denominator;
	return accuracy;

}

//k =  number of neighbors
void KNN_classifier_lab08(int k) {

	const int noOfBeans = 11;
	const int histSize = noOfBeans * 3;
	const int noOfInstances = 672;
	const int noOFTestInstances = 85;

	Mat y(noOfInstances, 1, CV_8UC1);
	Mat yTest(noOFTestInstances, 1, CV_8UC1);

	Mat X(noOfInstances, histSize, CV_32FC1);
	Mat XTest(noOFTestInstances, histSize, CV_32FC1);

	const int noOfClasses = 6;
	int fileNr = 0, rowX = 0;

	char classes[noOfClasses][10] =
	{ "beach", "city", "desert", "forest", "landscape", "snow" };

	int countTrain = 0;
	int countTest = 0;

	//read images from training set and add histogram vector to feature matrix
	char fname[256];
	char fnameTest[256];

	//load train instances
	for (int i = 0; i < noOfClasses; i++) {
		fileNr = 0;
		while (1) {
			sprintf(fname, "images_KNN/train/%s/%06d.jpeg", classes[i], fileNr++);
			Mat img = imread(fname);
			if (img.cols == 0) break;
			int* hist = colorHistogram(img, noOfBeans);
			//calculate the histogram in hist
			for (int d = 0; d < histSize; d++)
				X.at<float>(rowX, d) = hist[d];
			y.at<uchar>(rowX) = i;
			rowX++;
			countTrain++;
		}
	}

	rowX = 0;
	//load test instances
	for (int k = 0; k < noOfClasses; k++) {
		fileNr = 0;
		while (1) {
			sprintf(fnameTest, "images_KNN/test/%s/%06d.jpeg", classes[k], fileNr++);
			Mat img = imread(fnameTest);
			if (img.cols == 0) break;
			int* hist = colorHistogram(img, noOfBeans);
			//calculate the histogram in hist
			for (int d = 0; d < histSize; d++)
				XTest.at<float>(rowX, d) = hist[d];
			yTest.at<uchar>(rowX) = k;
			rowX++;
			countTest++;
		}
	}

	//confusion Matrix
	Mat C = Mat::zeros(noOfClasses, noOfClasses, CV_32FC1);
	// X - training set, y -label vector
	int testHist[histSize];
	for (int count = 0; count < noOFTestInstances; count++) {
		//obtain histogram for current image
		for (int d = 0; d < histSize; d++)
			testHist[d] = XTest.at<float>(count, d);
		//compute distances
		double distances[noOfInstances];
		for (int i = 0; i < noOfInstances; i++) {
			distances[i] = 0.0;
			for (int j = 0; j < histSize; j++) {
				distances[i] += (testHist[j] - X.at<float>(i, j))* (testHist[j] - X.at<float>(i, j));
			}
			distances[i] = sqrt(distances[i]);
		}
		//sort the distances
		double sortedDistances[noOfInstances];
		std::copy(distances, distances + noOfInstances, sortedDistances);
		std::sort(sortedDistances, sortedDistances + noOfInstances);
		//find the K nearest distances
		int voteHist[noOfClasses];
		for (int i = 0; i < noOfClasses; i++) {
			voteHist[i] = 0;
		}
		for (int i = 0; i < k; i++) {
			double currentDistance = sortedDistances[i];
			for (int j = 0; j < noOfInstances; j++) {
				if (currentDistance == distances[j]) {
					voteHist[(int)y.at<uchar>(j)] += 1;
				}
			}
		}
		int foundClass = -1;
		int max = -1;
		for (int idx = 0; idx < noOfClasses; idx++) {
			if (max < voteHist[idx]) {
				foundClass = idx;
				max = voteHist[idx];
			}
		}

		//update confusion matrix
		C.at<float>(foundClass, (int)yTest.at<uchar>(count)) += 1.0;
		std::cout << classes[foundClass] << " " << classes[yTest.at<uchar>(count)] << std::endl;

	}
	double accuracy = getAccuracyFromConfusionMatrix(C);
	std::cout << "Accuracy: " << accuracy << std::endl;

	getchar();
	waitKey();
	getchar();
}

//binarization of a grayscale image
Mat grayToBinary(Mat srcImg, int threshold) {
	Mat result = Mat::zeros(srcImg.rows, srcImg.cols, CV_8UC1);
	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if (srcImg.at<uchar>(i, j) < threshold) {
				result.at<uchar>(i, j) = 0;
			}
			else {
				result.at<uchar>(i, j) = 255;
			}
		}
	}
	return result;
}

/***************************************************BAYESIAN*****************************************************/
void bayesian_classifier_lab09() {

	const int featureSize = 28 * 28;
	const int noOfInstances = 3585;
	const int noOfClasses = 5;
	const int noOfTestInstances = 895;
	Mat X = Mat(noOfInstances, featureSize, CV_8UC1);
	Mat XTest = Mat(noOfTestInstances, featureSize, CV_8UC1);
	int rowX = 0;
	int rowXTest = 0;
	Mat y(noOfInstances, 1, CV_8UC1);
	Mat yTest(noOfTestInstances, 1, CV_8UC1);

	char classes[noOfClasses][10] =
	{ "0", "1", "2", "3", "4" };
	//load the train instances
	double priors[5];
	int elementsOfClassTrain[5];
	int priorNr;

	char fname[256];
	for (int i = 0; i < noOfClasses; i++) {
		priorNr = 0;
		while (1) {
			sprintf(fname, "images_Bayes/train/%s/%06d.png", classes[i], priorNr++);
			Mat img = imread(fname, CV_8UC1);
			if (img.cols == 0) break;
			Mat binary = grayToBinary(img, 150);

			int d = 0;
			for (int r = 0; r < binary.rows; r++) {
				for (int c = 0; c < binary.cols; c++) {
					X.at<uchar>(rowX, d) = binary.at<uchar>(r, c);
					d++;
				}
			}
			y.at<uchar>(rowX) = i;
			rowX++;
		}
		priors[i] = priorNr / (double)noOfInstances;
		elementsOfClassTrain[i] = priorNr;
	}

	//loatd the test instances
	char fnameTest[256];
	for (int i = 0; i < noOfClasses; i++) {
		switch (i) {
		case 0: priorNr = 784; break;
		case 1: priorNr = 908; break;
		case 2: priorNr = 827; break;
		case 3: priorNr = 808; break;
		case 4: priorNr = 258; break;
		}
		while (1) {
			sprintf(fnameTest, "images_Bayes/test/%s/%06d.png", classes[i], priorNr++);
			Mat img = imread(fnameTest, CV_8UC1);
			if (img.cols == 0) break;
			Mat binary = grayToBinary(img, 150);

			int d = 0;
			for (int r = 0; r < binary.rows; r++) {
				for (int c = 0; c < binary.cols; c++) {
					XTest.at<uchar>(rowXTest, d) = binary.at<uchar>(r, c);
					d++;
				}
			}
			yTest.at<uchar>(rowXTest) = i;
			rowXTest++;
		}
	}

	//compute likelihood
	//w/ laplace smoothing
	Mat likelihood = Mat::zeros(noOfClasses, featureSize, CV_64FC1);
	for (int k = 0; k < noOfInstances; k++) {
		for (int d = 0; d < featureSize; d++) {
			if (X.at<uchar>(k, d) == 255) {
				likelihood.at<double>((int)y.at<uchar>(k), d) += 1.0;
			}
		}
	}
	//laplace smoothing
	for (int r = 0; r < likelihood.rows; r++) {
		for (int c = 0; c < likelihood.cols; c++) {
			double value = likelihood.at<double>(r, c) + 1.0;
			likelihood.at<double>(r, c) = (value / (double)(noOfClasses + elementsOfClassTrain[r]));
		}
	}

	//classify test images
	Mat C = Mat::zeros(noOfClasses, noOfClasses, CV_32F);
	for (int count = 0; count < noOfTestInstances; count++) {
		Mat randImg = XTest.row(count);
		double classProbs[5];
		for (int c = 0; c < 5; c++) {
			classProbs[c] = log(priors[c]);
			for (int j = 0; j < featureSize; j++) {
				if (randImg.at<uchar>(0, j) == 255) {
					classProbs[c] += log(likelihood.at<double>(c, j));
				}
				else {
					classProbs[c] += log(1.0f - likelihood.at<double>(c, j));
				}
			}
		}
		double max = *std::max_element(classProbs, classProbs + 5);
		int predictedClass = -1;
		for (int i = 0; i < 5; i++) {
			if (max == classProbs[i]) {
				predictedClass = i;
			}
		}
		C.at<float>(predictedClass, yTest.at<uchar>(count)) += 1.0;
	}

	double accuracy = getAccuracyFromConfusionMatrix(C);
	std::cout << "Accuracy: " << accuracy << std::endl;

	waitKey(0);
	getchar();
	waitKey(0);
	getchar();

}

/***************************************************PERCEPTRON***************************************************/
void linearClassifies_perceptron_lab10() {
	Mat srcImg = imread("images_Perceptron/test06.bmp");
	int classes[] = { -1,1 };
	//scan the input image in order to obtain the dimensions of the 
	//training matrix X. each feature from X has 2 dimensions, the XY positions in the srcImg
	int rowX = 0;
	std::vector<std::pair<float, float>> features;
	std::vector<int> labels;
	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if (!(srcImg.at<Vec3b>(i, j)[0] == 255 && srcImg.at<Vec3b>(i, j)[1] == 255 && srcImg.at<Vec3b>(i, j)[2] == 255)) {
				rowX++;
				features.push_back(std::make_pair((float)i, (float)j));
				if (srcImg.at<Vec3b>(i, j)[0] == 255) {
					labels.push_back(1);
				}
				if (srcImg.at<Vec3b>(i, j)[2] == 255) {
					labels.push_back(-1);
				}

			}

		}
	}
	Mat X = Mat(rowX, 3, CV_64FC1);
	Mat y = Mat(rowX, 1, CV_64FC1);
	for (auto i = 0; i != features.size(); i++) {
		//augment each feature vector with a one
		X.at<double>(i, 0) = 1.0;
		X.at<double>(i, 1) = (double)features[i].first;
		X.at<double>(i, 2) = (double)features[i].second;
		y.at<double>(i) = (double)labels[i];
	}
	double niu = 0.0001;
	double w[3] = { 0.1,0.1,0.1 };
	double ELimit = 0.00001;
	int maxIter = 100000;

	for (int i = 0; i < maxIter; i++) {
		double E = 0.0;
		for (int j = 0; j < rowX; j++) {
			double z = 0.0;
			for (int d = 0; d < 3; d++) {
				z += w[d] * X.at<double>(j, d);
			}
			if (z * y.at<double>(j) <= 0) {
				for (int c = 0; c < 3; c++) {
					w[c] += niu * X.at<double>(j, c) * y.at<double>(j);
				}
				E += 1.0f;
			}
		}
		E = E / (double)rowX;
		if (E < ELimit)
			break;
	}
	//find endpoints for the line
	Point2d one(1.0, 0.0);
	one.x = -w[0] / w[2];

	Point2d two(0.0, 1.0);
	two.y = -w[0] / w[1];
	fullLine(srcImg, one, two, Scalar(0, 255, 0));
	imshow("title", srcImg);
	waitKey(0);
	//w0+w1y+w2x
}

/***************************************************ADABOOST*****************************************************/
#define MAXT 1000
struct weaklearner {
	int feature_i;
	int threshold;
	int class_label;
	float error;
	//here X, i think, it's a row(sample) from the big training set X
	int classify(Mat X) {
		if (X.at<double>(feature_i)<threshold)
			return class_label;
		else
			return -class_label;
	}
};

struct classifier {
	int T;
	float alphas[MAXT];
	weaklearner hs[MAXT];
	int classify(Mat X) {
		int sum = 0;
		for (int i = 0; i < T; i++) {
			sum += hs[i].classify(X);
		}
		return (sum > 0) - (sum < 0);
	}
};

weaklearner findWeakLearner(Mat X, Mat y, Mat w, int imgSize) {
	weaklearner bestH;
	int classLabels[2] = { -1, 1 };
	double bestError = FLT_MAX;
	std::vector<double> z(X.rows);
	for (int j = 0; j < X.cols; j++) {
		for (int treshold = 0; treshold < imgSize; treshold++) {
			for (int c = 0; c < 2; c++) {
				double error = 0.0f;
				for (int i = 0; i < X.rows; i++) {
					if (X.at<double>(i, j) < treshold) {
						z.at(i) = (classLabels[c]);
					}
					else {
						z.at(i) = (-classLabels[c]);
					}
					if (z.at(i)*y.at<double>(i) < 0) {
						error += w.at<double>(i);
					}
				}
				if (error < bestError) {
					bestError = error;
					bestH.threshold = treshold;
					bestH.error = error;
					bestH.class_label = classLabels[c];
					bestH.feature_i = j;
				}
			}
		}
	}
	return bestH;

}

void drawBoundary(Mat srcImg, classifier clf) {
	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if ((srcImg.at<Vec3b>(i, j)[0] == 255 && srcImg.at<Vec3b>(i, j)[1] == 255 && srcImg.at<Vec3b>(i, j)[2] == 255)) {
				Mat X = Mat(1, 2, CV_64FC1);
				X.at<double>(0, 0) = (double)i;
				X.at<double>(0, 1) = (double)j;
				if (clf.classify(X) == 1) {
					srcImg.at<Vec3b>(i, j)[0] = 0;
					srcImg.at<Vec3b>(i, j)[1] = 255;
					srcImg.at<Vec3b>(i, j)[2] = 255;
				}
				else {
					srcImg.at<Vec3b>(i, j)[0] = 208;
					srcImg.at<Vec3b>(i, j)[1] = 224;
					srcImg.at<Vec3b>(i, j)[2] = 64;

				}
			}
		}
	}
}

void adaBoost_lab11(int T) {

	Mat srcImg = imread("images_AdaBoost/points3.bmp");
	int classes[] = { -1, 1 };
	//scan the input image in order to obtain the dimensions of the 
	//training matrix X. each feature from X has 2 dimensions, the XY positions in the srcImg
	int rowX = 0;
	std::vector<std::pair<float, float>> features;
	std::vector<int> labels;
	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if (!(srcImg.at<Vec3b>(i, j)[0] == 255 && srcImg.at<Vec3b>(i, j)[1] == 255 && srcImg.at<Vec3b>(i, j)[2] == 255)) {
				rowX++;
				features.push_back(std::make_pair((float)i, (float)j));
				if (srcImg.at<Vec3b>(i, j)[0] == 255) {
					labels.push_back(-1);
				}
				if (srcImg.at<Vec3b>(i, j)[2] == 255) {
					labels.push_back(1);
				}

			}

		}
	}
	Mat X = Mat(rowX, 2, CV_64FC1);
	Mat y = Mat(rowX, 1, CV_64FC1);
	Mat w = Mat(rowX, 1, CV_64FC1);
	for (auto i = 0; i != features.size(); i++) {
		//augment each feature vector with a one
		X.at<double>(i, 0) = (double)features[i].first;
		X.at<double>(i, 1) = (double)features[i].second;
		y.at<double>(i) = (double)labels[i];
		w.at<double>(i) = 1.0 / rowX;
	}
	std::vector<double>alfa;
	classifier adaBoostClassifier;
	for (int t = 0; t < T; t++) {
		weaklearner h = findWeakLearner(X, y, w, srcImg.rows);
		alfa.push_back(0.5* (log((1 - h.error) / h.error)));
		double s = 0.0;

		for (int i = 0; i < rowX; i++) {
			//wrongly classified examples will have 𝒚𝒊𝒉𝒕(𝑿𝒊) < 𝟎
			//their weights will be higher in the next step
			double newW = w.at<double>(i) * exp(-alfa.at(t) * y.at<double>(i) *h.classify(X.row(i)));
			w.at<double>(i) = newW;
			s += w.at<double>(i);
		}

		for (int i = 0; i < rowX; i++) {
			double normalizedW = w.at<double>(i) / s;
			w.at<double>(i) = normalizedW;
		}
		adaBoostClassifier.alphas[t] = alfa.at(t);
		adaBoostClassifier.T = T;
		adaBoostClassifier.hs[t] = h;
	}

	drawBoundary(srcImg, adaBoostClassifier);
	imshow("result", srcImg);
	waitKey(0);
}

/************************************************************* P R S E N D*************************************************************************/

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - LinearRegression - Line Fitting\n");
		printf(" 11 - RANSAC - Line fitting \n");
		printf(" 12 - Hough Transform\n");
		printf(" 13 - Distance Transform Pattern Matching\n");
		printf(" 14 - Statistical Analysis - Faces\n");
		printf(" 15 - K Means Clunstering - Set of Points\n");
		printf(" 16 - K Means Clunstering - GrayScale\n");
		printf(" 17 - K Means Clustering - RGB\n");
		printf(" 18 - Principal Component Analysys\n");
		printf(" 19 - KNN Classifier\n");
		printf(" 20 - Naive Bayesian Classifier\n");
		printf(" 21 - Percepton\n");
		printf(" 22 - Adaboost\n");


		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			linearRegression_Lab01();
			break;
		case 11:
			RANSAC_lab02();
			break;
		case 12:
			HoughTransform_lab03();
			break;
		case 13:
			patternMatchingWithDT_lab04();
			break;
		case 14:
			statisticalAnalysis_lab05();
			break;
		case 15:
			KMeansClustering_labo06_setOfPoints(2);
			break;
		case 16:
			KMeansClustering_lab06_grayScale(2);
			break;
		case 17:
			KMeansClustering_lab06_RGB(2);
			break;
		case 18:
			principalComponentAnalysis_lab07();
			break;
		case 19:
			KNN_classifier_lab08(7);
			break;
		case 20:
			bayesian_classifier_lab09();
			break;
		case 21:
			linearClassifies_perceptron_lab10();
			break;
		case 22:
			adaBoost_lab11(7);
			break;
		}
	} while (op != 0);
	return 0;
}