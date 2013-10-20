#ifndef __CURVE_DETECT_H
#define __CURVE_DETECT_H

#include <cv.h>

using namespace cv;

namespace cd {
    class CurveDetect {
    private:
        int boxWidth, boxHeight;
        
        double findBestTheta(const Mat& roi, Point center, int radius);
        void canny(const Mat& src, Mat& edges);
        void rotate(const cv::Mat& src, Point2f center, double angleRad, cv::Mat& dst);
        Rect getROIRect(const Point2f& pt);
        void getPointsOfInterest(const Mat& roi, Mat& points);
        Mat polypoint(const Mat& poly, float x);
        
    public:
        CurveDetect(int boxHeight, int boxWidth);

        void fitCurve(const Mat& src, const Point2f& startPt, int maxCurves);
        
    };
}
#endif //__CURVE_DETECT_H