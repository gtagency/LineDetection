
#include "curve_detect.h"

#include "math.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include "curve_detect.h"

using namespace cd;
using namespace cv;

// #define DEBUG

Scalar LOW_HSV = Scalar(60, 50, 50);
Scalar HIGH_HSV = Scalar(90, 255, 255);
void getBinary(Mat& src, Scalar& low_HSV, Scalar& hi_HSV, Mat& dest) {
    Mat frame = src.clone();
    cvtColor(frame, frame, CV_BGR2HSV);
    Mat bw;
    inRange(frame, low_HSV, hi_HSV, bw);
    // vector<vector<Point> > contours;
    // findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    //
    // dest = Mat::zeros(bw.size(), bw.type());
    // drawContours(dest, contours, -1, Scalar::all(255), CV_FILLED);
    dest = bw;
}

int main(int argc, const char **argv) {
#if 0
    Mat raw = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat src;
    getBinary(raw, LOW_HSV, HIGH_HSV, src);
#else
    Mat src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
#endif
    CurveDetect cd = CurveDetect(40, 20);
    Point start = Point(350, 705);
    cd.fitCurve(src, start, 15);
    
    Mat show = src.clone();
    std::vector<Point>::iterator it = cd.getPointsOnCurve().begin();
    Point pt = *it;
    while (++it != cd.getPointsOnCurve().end()) {
        Point nextPt = *it;
        printf ("Line from (%d, %d) to (%d, %d)\n", pt.x, pt.y, it->x, it->y);
        line(show, pt, nextPt, Scalar(127, 127, 127));
	pt = nextPt;
    }

    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", show );                   // Show our image inside it.
    waitKey(0);
    
    
}
