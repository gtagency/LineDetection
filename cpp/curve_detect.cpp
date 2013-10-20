
#include "curve_detect.h"

#include "math.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace cd;
using namespace cv;

CurveDetect::CurveDetect(int bh, int bw)
    : boxHeight(bh), boxWidth(bw) {
    
}

/*
const int channels = I.channels();
    switch(channels)
    {
    case 1:
        {
            for( int i = 0; i < I.rows; ++i)
                for( int j = 0; j < I.cols; ++j )
                    I.at<uchar>(i,j) = table[I.at<uchar>(i,j)];
            break;
        }
    case 3:
        {
         Mat_<Vec3b> _I = I;

         for( int i = 0; i < I.rows; ++i)
            for( int j = 0; j < I.cols; ++j )
               {
                   _I(i,j)[0] = table[_I(i,j)[0]];
                   _I(i,j)[1] = table[_I(i,j)[1]];
                   _I(i,j)[2] = table[_I(i,j)[2]];
            }
         I = _I;
         break;
        }
    }

    return I;
    */

double CurveDetect::findBestTheta(const Mat& roi, Point center, int radius) {
    double theta;
    double twoPi = (2 * M_PI);
    bool found = false;
    //find the first "theta" by sweeping the region of interest for the first non zero pixel
    for (int ii = 0; ii < 360; ii++) {
        int x = boxWidth/2 + radius * cos(ii / twoPi);
        int y = boxHeight/2 + radius * sin(ii / twoPi);
        printf("%d", roi.at<uchar>(y,x));
        if (roi.at<uchar>(y,x) != 0) {
            theta = ii / twoPi;
            found = true;
            break;
        }
    }
    if (!found) {
        return -1;
    }
    return theta;
}

void CurveDetect::canny(const Mat& src, Mat& edges) {
    //TODO: may have to get clever about selecting threshold
    int edgeThresh = 1;
    double low_thres = 30; //120;
    int const max_lowThreshold = 200;
    int ratio = 2.7;
    int kernel_size = 3;
    cv::Mat detected_edges;
    
    Mat junk;
    // double high_thres = cv::threshold( src, junk, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU ) ;
    // printf("high_thres: %f\n", high_thres);
    // lowThreshold = high_thres * 0.5;
    /// Reduce noise with a kernel 3x3
    double high_thres = low_thres*ratio;
    cv::blur( src, detected_edges, cv::Size(3,3) );
    /// Canny detector
    cv::Canny( detected_edges, detected_edges, low_thres, low_thres, kernel_size );

    /// Using Canny's output as a mask, we display our result
    edges = cv::Scalar::all(0);
    printf("Detected Size: %d %d\n", detected_edges.rows, detected_edges.cols);

    src.copyTo( edges, detected_edges);
    // edges = src;
}

/**
 * Rotate an image
 */
void CurveDetect::rotate(const cv::Mat& src, Point2f center, double angleRad, cv::Mat& dst) {
    int len = std::max(src.cols, src.rows);
    cv::Mat r = cv::getRotationMatrix2D(center, angleRad * 180.0/M_PI, 1.0);

    cv::warpAffine(src, dst, r, cv::Size(len, len));
}

Rect CurveDetect::getROIRect(const Point2f& pt) {
    //TODO: may have to be negative or something
    float x = fmax(0, pt.x - boxWidth/2);
    float y = fmax(0, pt.y - boxHeight);
    printf("%f, %f, %d, %d\n", x, y, boxWidth, boxHeight);
    return Rect(x, y, boxWidth, boxHeight);
}

//points is a n x 2 matrix, with x in first column and y in the second column
void CurveDetect::getPointsOfInterest(const Mat& roi, Mat& points) {
    Mat edges;
    canny(roi, edges);
    Mat locations;   // output, locations of non-zero pixels 
    //locations should be a n x 1 vector of points
    findNonZero(edges, locations);
    printf("Found edges: %d, %d\n", locations.rows, locations.cols);
    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
        imshow( "Display window", edges );                   // Show our image inside it.
        waitKey(0);
    //adjust the points so they're relative to axes at the bottom center of the box.
    points = Mat::zeros(locations.rows, 2, CV_32FC1);
    for (int r = 0; r < locations.rows; r++) {
        Point *locPtr = locations.ptr<Point>(r);
        float *ptPtr  = points.ptr<float>(r);
        printf("%d, %d\n", locPtr[0].x, locPtr[0].y);
        ptPtr[0] = locPtr[0].x;
        ptPtr[1] = locPtr[0].y;
    } 
}

Mat CurveDetect::polypoint(const Mat& poly, float x) {
    //TODO: currently only works with 1 degree polynomial...can expand to n degree using varargs
    printf("Poly: %f, %f, x: %f\n", poly.at<float>(0), poly.at<float>(1), x);
    return (Mat_<float>(2,1) << x, poly.at<float>(0) + x * poly.at<float>(1));
}

//NOTE: assumes src is a binary image
void CurveDetect::fitCurve(const Mat& src, const Point2f& startPt, int maxCurves) {
    int curves = 0;
    Size size = src.size();
    printf("Source Size: %d %d\n", src.rows, src.cols);
    //TODO: preprocess image? convert color, etc
    printf("Detecting best initial theta\n");
    Mat roi = Mat(src, getROIRect(startPt));
    //NOTE: this assumes that the curve is at least as big as one box in width and length
    // also assumes width, height >= 2
    double theta0 = M_PI/2.7;//findBestTheta(roi, Point(boxWidth/2, boxHeight/2), fmin(boxWidth/2, boxHeight/2));
    printf("Best theta: %.02f", theta0);
    printf("Start curve fitting\n");
    std::vector<Point> points;

    Mat A = (Mat_<float>(2,2) << 1, 0, 0, -1);

    // Mat can;
    // canny(src, can);
    
    // namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    //     imshow( "Display window", can );                   // Show our image inside it.
    // 
    //     waitKey(0);
    points.push_back(startPt);
    Point2f pt = startPt;
    Mat show = src.clone();
    
    double theta = theta0;
    while (pt.x < size.width && pt.y < size.height && curves < maxCurves) {
        printf("%f\n", theta);
        float rotation = M_PI/2 - theta;
        printf("Rotate image: %f radians\n", rotation);
        Mat rot;
        rotate(src, pt, rotation, rot);
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
            imshow( "Display window", rot );                   // Show our image inside it.
            waitKey(0);
        printf("Extract ROI\n");
        roi = Mat(rot, getROIRect(pt));
        printf("Get points of interest\n");
        Mat points;
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
            imshow( "Display window", roi );                   // Show our image inside it.
            waitKey(0);
        getPointsOfInterest(roi, points);
        // Mat edges;
        // canny(roi, edges);
        // Mat locations;   // output, locations of non-zero pixels 
        // findNonZero(edges, locations);
        // printf("Found edges: %d, %d\n", locations.rows, locations.cols);
        //     
        // //adjust the points so they're relative to axes at the bottom center of the box.
        // std::vector<float> xs;
        // std::vector<float> ys;
        // for (int r = 0; r < locations.rows; r++) {
        //     Point *ptr = locations.ptr<Point>(r);
        //     for (int c = 0; c < locations.cols; c++) {
        //         printf("%d, %d\n", ptr[c].x, ptr[c].y);
        //         xs.push_back(ptr[c].x - boxWidth / 2);
        //         ys.push_back(ptr[c].y - boxHeight);
        //     }
        // } 
        
        // for (int ii = 0; ii < xs.size(); ii++) {
        //     printf("%f, %f\n", xs[ii], ys[ii]);
        // }
        // 
        
        //fit a y,x polynomial, so we can compute x values for the top and bottom
        // of the box
        Mat poly;
        Mat xsMat = points.col(0) - boxWidth/2;
        Mat ysMat = points.col(1) - boxHeight;
        poly = Mat::zeros(2, 1,CV_32FC1);
        printf("%d, %d, %d, %d, %d, %d, %d, %d, %d\n", xsMat.rows, xsMat.cols, xsMat.type(), ysMat.rows, ysMat.cols, ysMat.type(), poly.rows, poly.cols, poly.type());
        for (int r = 0; r < xsMat.rows; r++) {
            printf("%f, %f\n", xsMat.ptr<float>(r)[0], ysMat.ptr<float>(r)[0]);
        } 
        polyfit(ysMat, xsMat, poly, 1);
        //so we're in order c(0) + c(1)x + c(2)x^2...
        // flip(poly, poly, -1);
        printf("Polynomial: %f, %f\n", poly.at<float>(0), poly.at<float>(1));
        
        //these points are y,x
        Mat p1 = polypoint(poly, -boxHeight);
        Mat p2 = polypoint(poly, 0);

        printf("DEBUG testing the poly points\n");
        printf("%f, %f\n", p1.at<float>(0), p1.at<float>(1));
        printf("%f, %f\n", p2.at<float>(0), p2.at<float>(1));

        Mat Rt = (Mat_<float>(2,2) << cos(rotation), -sin(rotation),
                                      sin(rotation),  cos(rotation));
        Rt = Rt.t();
                                  
        // printf("%f, %f\n", x1, x2);
        flip(p1, p1, -1);
        flip(p2, p2, -1);
        Mat ptMat = (Mat_<float>(2,1) << pt.x, pt.y);
        p1 = A * Rt * A * p1 + ptMat;
        p2 = A * Rt * A * p2 + ptMat;

        printf("%f, %f\n", p1.at<float>(0), p1.at<float>(1));
        printf("%f, %f\n", p2.at<float>(0), p2.at<float>(1));

        Mat nextPt = p1;
        if (norm(ptMat - p1) < norm(ptMat - p2)) {
            nextPt = p2;
        }

        printf("%f, %f\n", pt.x, pt.y);
        printf("%f, %f\n", nextPt.at<float>(0), nextPt.at<float>(1));
        
        Point pt2 = Point(nextPt.at<float>(0), nextPt.at<float>(1));
        line(show, pt, pt2, Scalar(127, 127, 127));
        namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
            imshow( "Display window", show );                   // Show our image inside it.
        pt = pt2;
        printf("%f, %f\n", atan(poly.at<float>(1)), theta);
        theta = atan(poly.at<float>(1)) + theta;
        printf("%f, %f\n", pt.x, pt.y);
        // dst = W.clone(); do cv::flip(W,dst,-1);
        curves++;
        waitKey(0);
    }
//     points = zeros(2,4);
//     thetas = zeros(1,4);
//     p = p0;
//     theta = theta0;
// 
//     for i=1:num
//         points(:,i) = p;
//         thetas(i) = theta;
//         [p theta] = aoi_func(img, p, theta, aoi_width, aoi_height);
//     end
//     points(:,i+1) = p;
//     thetas(i+1) = theta;
}

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
    Mat src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat bin = src;
    // getBinary(src, LOW_HSV, HIGH_HSV, bin);
    
    CurveDetect cd = CurveDetect(40, 20);
    Point2f start = Point2f(350, 705);
    cd.fitCurve(bin, start, 15);
}