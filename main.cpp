#include "main.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

#include <iostream>
#include <cmath>
#include <ctype.h>
#include <stdio.h>
#include <sys/stat.h>

// #include "MXLines.hpp"
#include "RasterSpace.hpp"

using namespace cv;
using namespace std;

double scale = 0.5;

Ptr<BackgroundSubtractorMOG2> backsub = createBackgroundSubtractorMOG2(100, 7, false);

Mat kernel_close = getStructuringElement(MORPH_ELLIPSE, Size(7, 7), Point(-1, -1));
Mat kernel_open = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));
Mat kernel_dilate  = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1));

int framenum = 0;

int main( int argc, char** argv )
{
    if (! check_argv(argc, argv)){
        return 1;
    }
    const string videoPath = string(argv[1]);
    run(videoPath);

    return 0;
}

void run(const string videoPath) {

    VideoCapture cap = VideoCapture(videoPath);
    int fps = int(cap.get(CAP_PROP_FPS));
    int width = int(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = int(cap.get(CAP_PROP_FRAME_HEIGHT));
    float area = width * height * scale * scale;
    float mask_thr = area * 0.06;

    double scaleSpaceSize = 0.5;
    const int SpaceSize = int(height * scaleSpaceSize);
    const int searchRange = SpaceSize / 20;
    const int margin = 0;

    const int houghTransfromThreshold = 90;
    const float minAngle = CV_PI/3;
    const float maxAngle = CV_PI/1.8;


    int psize = 14;
    int SubPixelRadius = 2;
    float Normalization = 1;

    cout<<"Space Size: "<<SpaceSize<<endl;

    uint** pSpace = new uint *[SpaceSize];
    for(int i = 0; i <SpaceSize; i++)
        pSpace[i] = new uint[SpaceSize];

    for(int i = 0; i <SpaceSize; i++)
        for(int j = 0; j <SpaceSize; j++)
            pSpace[i][j] = 0;


    float w_c = (width * scale - 1)/2.f;
    float h_c = (height * scale - 1)/2.f;
    float norm = (max(w_c, h_c) - 14);


    while(true)
    {
        Mat frame;
        cap >> frame;

        if( frame.empty() )
        {
            if (videoPath.find("rtsp://") != std::string::npos)
            {
                cap.release();
                cap = VideoCapture(videoPath);
                continue;
            }
            else
                break;
        }

        Mat frame_rs;
        resize(frame, frame_rs, Size(0,0), scale, scale, INTER_LINEAR);
        Mat mask, maskBigger;
        backsub->apply(frame_rs, mask, 0.01);


        if (framenum > 30 && countNonZero(mask) > mask_thr)
        {
            // cout<<endl;
            morphologyEx(mask, mask, MORPH_OPEN, kernel_open);
            // morphologyEx(mask, mask, MORPH_CLOSE, kernel_close);
            morphologyEx(mask, mask, MORPH_DILATE, kernel_dilate);

            morphologyEx(mask, maskBigger, MORPH_DILATE, kernel_dilate);

            Mat edges, edges2, gray , output, cdst;
            frame_rs.copyTo(output, maskBigger);
            cvtColor(output, gray, COLOR_BGR2GRAY);

            Canny(gray, edges, 100, 200);
            edges.copyTo(edges2, mask);

            cvtColor(edges2, cdst, CV_GRAY2BGR);

            vector<Vec2f> lines;
            list<line_param> mxlines;

            // detect lines
            HoughLines(edges2, lines, 1, CV_PI/180, houghTransfromThreshold, 0, 0, minAngle, maxAngle);

            // draw lines
            for( size_t i = 0; i < lines.size(); i++ )
            {
                float rho = lines[i][0], theta = lines[i][1];
                Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a*rho, y0 = b*rho;
                pt1.x = cvRound(x0 + 1000*(-b));
                pt1.y = cvRound(y0 + 1000*(a));
                pt2.x = cvRound(x0 - 1000*(-b));
                pt2.y = cvRound(y0 - 1000*(a));
                line( cdst, pt1, pt2, Scalar(0,0,255), 1, CV_AA);

                // cout<<"line: "<<pt1.x<<", "<<pt1.y<<", "<<pt2.x<<", "<<pt2.y<<endl;


                // normalize lines
                float aa=pt2.y-pt1.y;
                if(aa==0) continue;
                float bb=pt1.x-pt2.x;
                float t=sqrt(aa*aa+bb*bb);
                // if(t<35) continue;
                aa=aa/t;
                bb=bb/t;
                float c=0-(pt1.y-h_c)*bb-(pt1.x-w_c)*aa;

                mxlines.push_back(line_param(aa,bb,c/norm, 1)); // t*1.f - weight
            }






            // list<line_param> lines = get_mx_lines(edges2, psize, Normalization);
            Point2f cc_Vanp = calc_CC_Vanp(pSpace, mxlines, SpaceSize, Normalization, edges.rows, edges.cols, SubPixelRadius, searchRange, margin, 2);
            printf("%f, %f \n", cc_Vanp.x, cc_Vanp.y);


            for(int y = 0; y < frame_rs.rows; y += frame_rs.rows/15)
            {
                if (cc_Vanp.x > 0)
                    line(frame_rs, cc_Vanp, Point(0, y), Scalar(0,255,0), 1);
                else
                    line(frame_rs, cc_Vanp, Point(frame_rs.cols, y), Scalar(0,255,0), 1);
            }


            // Draw accumulator
            int i, j;
            int mx = 0;
            for (i = 0; i < SpaceSize; i++)
                for (j = 0; j < SpaceSize; j++)
                    if (mx < pSpace[i][j])
                        mx = pSpace[i][j];

            Mat accum = Mat::zeros(SpaceSize, SpaceSize, CV_8UC1), accumBGR;

            for (i = 0; i < SpaceSize; i++)
                for (j = 0; j < SpaceSize; j++)
                    accum.at<uchar>(i,j) = int((double)pSpace[i][j] * 255 / mx);

            cvtColor(accum, accumBGR, CV_GRAY2BGR);

            Point2f PC_VanP = find_maximum(pSpace, SpaceSize, SubPixelRadius, searchRange, margin, 2);
            // cout<<PC_VanP.y<<", "<<PC_VanP.x<<endl;

            // draw founded maximum
            int pdd = 2;
            rectangle(accumBGR, Point(PC_VanP.x-pdd, PC_VanP.y-pdd), Point(PC_VanP.x+pdd, PC_VanP.y+pdd), Scalar(0,0,255));

            // draw searching area
            // top rectangle
            rectangle(accumBGR, Point(int(SpaceSize/2 - searchRange/2), margin), Point(int(SpaceSize/2 + searchRange/2), searchRange), Scalar(0,255,0));
            // bottom rectangle
            rectangle(accumBGR, Point(int(SpaceSize/2 - searchRange/2), SpaceSize - searchRange), Point(int(SpaceSize/2 + searchRange/2), SpaceSize-margin), Scalar(0,255,0));

            imshow("Original", frame_rs);
            imshow("Mask", output);
            imshow("Accumulator", accumBGR);
            imshow("detected lines", cdst);

            int ch = waitKey(1);
            if (ch == 27)
             break;

        }
        framenum += 1;

        // if (framenum > 200)
        // {
        //  for(int i = 0; i <SpaceSize; i++)
        //         delete [] pSpace[i];
        //     delete [] pSpace;

        //     destroyAllWindows();
        //  exit(0);
        // }
    }

    for(int i = 0; i <SpaceSize; i++)
        delete [] pSpace[i];
    delete [] pSpace;

    destroyAllWindows();
    cap.release();

}


bool check_argv(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " FILEPATH.mp4" << endl;
        return false;
    }
    const char* videoPath = argv[1];
    struct stat sb;
    stat(videoPath, &sb);
    if (!S_ISREG(sb.st_mode)) {
        std::cerr << "Error, file does not exists: " << videoPath << endl;
        return false;
    }
    return true;
}
