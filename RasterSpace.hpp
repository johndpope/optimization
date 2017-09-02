#ifndef RASTERSPACE_H
#define RASTERSPACE_H

#include <iostream>
#include <cmath>
#include <ctype.h>
#include <stdio.h>

using namespace cv;
using namespace std;

struct line_param
{
    float a;
    float b;
    float c;
    float w;
    line_param(float _a,float _b, float _c, float _w):a(_a),b(_b),c(_c),w(_w){}
};


template <typename T> int sgn(T val) 
{
    return (T(0) <= val) - (val < T(0));
}

template <typename T> int sign(T val) 
{
    return (T(0) <= val) - (val <= T(0));
}

Point2f PC_point_to_CC(Point2f NormVP, float Normalization, int height, int width)
{
    float x = NormVP.x;
    float y = NormVP.y;

    // if(abs(x) < 0.005)
    //     return Point2f(0,0);

    int m = max(height, width);
    // printf("Wid: %d, Hei: %d, Max: %d\n", width, height, m);

    float v1 = y/x;
    float w2 = (sign(y) * y + sign(x) * x - 1) / x;
    float u3 = 1.0;
    // printf("v1: %f, w2: %f, u3: %f \n",v1,w2,u3);

    return Point2f((v1/Normalization*(m-1)+width+1)/2, (w2/Normalization*(m-1)+height+1)/2);
}

Point2f normalize_PC_points(Point2f VP, int spaceSize)
{
    return Point2f((2 * VP.x - (spaceSize + 1)) / (spaceSize - 1), (2 * VP.y - (spaceSize + 1)) / (spaceSize - 1));
}

Point2f find_maximum(uint ** space, int spaceSize, int R, int searchRange, int margin, int vp = 1)
{
    int xx = 0, yy = 0, mx = 0;

    if (vp == 1)
    {
        for (int i = spaceSize/2 - 5; i < spaceSize/2 + 5; i++)
           for (int j = spaceSize/2 - 5; j < spaceSize/2 + 5; j++)
                space[i][j] = 0;

        for (int i = margin; i < spaceSize - margin; i++)
            for (int j = margin; j < spaceSize - margin; j++)
                if (mx < space[i][j])
                {
                    mx = space[i][j];
                    yy = i+1;
                    xx = j+1;
                }
    }
    else
    {
        // search in top rectangle
        for (int i = margin; i < searchRange; i++)
            for (int j = int(spaceSize/2 - searchRange/2); j < int(spaceSize/2 + searchRange/2); j++)
                if (mx < space[i][j])
                {
                    mx = space[i][j];
                    yy = i+1;
                    xx = j+1;
                }

        // search in bottom rectangle
        for (int i = spaceSize - searchRange; i < spaceSize - margin; i++)
            for (int j = int(spaceSize/2 - searchRange/2); j < int(spaceSize/2 + searchRange/2); j++)
                if (mx < space[i][j])
                {
                    mx = space[i][j];
                    yy = i+1;
                    xx = j+1;
                }
    }

    // printf("Xi in arr: %d, Yi in arr: %d \n", xx, yy);

    int oSize = 2 * R + 1;
    int **O = (int **)calloc(oSize,sizeof(double));
    for(int i = 0; i < oSize; i++)
        O[i] = (int *)calloc(oSize,sizeof(double));

    for(int i = yy - R, ii = 0; i <= yy + R; i++, ii++)
        for(int j = xx - R, jj = 0; j <= xx + R; j++, jj++)
            if(i < 0 || i >= spaceSize || j < 0 || j >= spaceSize)
                O[ii][jj] = 0;
            else
                O[ii][jj] = space[i][j];

    float sumSR = 0, sumSC = 0, sumO = 0;
    for(int i = -R, ii = 0; i <= R; i++, ii++)
        for(int j = -R, jj = 0; j <= R; j++, jj++)
        {
            sumSR += O[ii][jj] * i;
            sumSC += O[ii][jj] * j;
            sumO += O[ii][jj];
        }

    for(int i = 0; i < oSize; i++)
        free(O[i]);
    free(O);

    return Point2f(xx + sumSC/sumO, yy + sumSR/sumO);
}

inline void lineH(int x0, int y0, int x1, int y1, uint ** space, int weight)
{
    float slope = (float)(y1 - y0)/(x1 - x0);

    float y_start = y0 + 0.5f; 
    float y_iter = y_start;
    
    int step = (x0 < x1) ? 1 : -1;
    slope *= step;
    
    for(int x = x0, c = 1; x != x1; x+=step, c++)
    {   
        space[x][int(y_iter)] += weight;        
        y_iter = y_start + c*slope;
    }
    
}

inline void lineV(int x0, int y0, int x1, int y1, uint ** space, int weight)
{
    // printf("%d, %d, %d, %d \n", x0, y0, x1, y1);

    float slope = (x1 - x0)/(float)(y1 - y0);

    float x_start = x0 + 0.5f; 
    float x_iter = x_start;
    int step = (y0 < y1) ? 1 : -1;
    slope *= step;

    // printf("%f, %f, %f \n", slope, x_start, x_iter);
    for(int y = y0, c = 1; y != y1; y+=step, c++)
    {   
        space[int(x_iter)][y] += weight;
        // printf("%d, %d \n", int(x_iter), y);
        x_iter = x_start + c*slope;        
    }     
}

void rasterize_lines(list<line_param> lines, int * endpoints, uint** space, int cSpaceSize, int numLines)
{ 
    int k = 0;
    for (list<line_param>::iterator i = lines.begin(); i!=lines.end(); ++i)
    {
        int * end = endpoints + k*8; 
        k++;
        int weight = i->w;

        for(int j=0; j<6; j+=2)
        {
            // cout<<(abs(end[j+3] - end[j+1]) > abs(end[j+2] - end[j]))<<endl;
            if(abs(end[j+3] - end[j+1]) > abs(end[j+2] - end[j]))
                lineV(end[j], end[j+1], end[j+2], end[j+3], space, weight);
            else
                lineH(end[j], end[j+1], end[j+2], end[j+3], space, weight);
        }        
        space[end[7]][end[6]] += weight;
    }
}

void lines_end_points(list<line_param> lines, int * endpoints, float space_c, int numLines)
{
    int center = round(space_c);

    int j = 0;
    for (list<line_param>::iterator i = lines.begin(); i!=lines.end(); ++i)
    {
        float a = i->a;
        float b = i->b; 
        float c = i->c;

        // printf("%f, %f, %f \n", a, b, c);

        float alpha = float(sgn(a*b));
        float beta = float(sgn(b*c));
        float gamma = float(sgn(a*c));
        
        // printf("%f, %f, %f \n", alpha, beta, gamma);

        int * end = endpoints + j*8;
        j++;

        float a_x = alpha*a / (c + gamma*a);
        float b_x = -alpha*c / (c + gamma*a);

        // printf("%f, %f \n", a_x, b_x);

        end[1] = round((a_x + 1) * space_c);
        end[0] = round((b_x + 1) * space_c);

        end[3] = round((b / (c + beta*b) + 1) * space_c);
        end[2] = center;

        end[5] = center;
        end[4] = round((b / (a + alpha*b) + 1) * space_c);

        end[7] = round((-a_x + 1) * space_c);
        end[6] = round((-b_x + 1) * space_c); 
    }
}

Point2f calc_CC_Vanp(uint** space, list<line_param> lines, int SpaceSize, float Normalization, int height, int width, int SubPixelRadius, int searchRange, int margin, int vp)
{  

    float space_c = (SpaceSize - 1.f)/2;

    int numLines = lines.size();
    int * EndPoints =  (int*) malloc(sizeof(int)*8*numLines);

    //Get all EndPoints
    lines_end_points(lines, EndPoints, space_c, numLines);

    // cout<<endl<<endl<<"End points"<<endl;
    // for (int k = 0; k < lines.size(); ++k)
    // {
    //     int * end = EndPoints + k*8;
    //     for(int j=0; j<8; ++j)
    //         cout<<end[j]<<", ";

    //     cout<<endl;
    // }

    //Rasterize
    rasterize_lines(lines, EndPoints, space, SpaceSize, numLines);
    
    free(EndPoints);


    Point2f PC_VanP = find_maximum(space, SpaceSize, SubPixelRadius, searchRange, margin, vp);
    // printf("PC_VanP: %f, %f\n", PC_VanP.x, PC_VanP.y);


    Point2f PC_NormVP = normalize_PC_points(PC_VanP, SpaceSize);
    // printf("PC_NormVP: %f, %f\n", PC_NormVP.x, PC_NormVP.y);

    Point2f CC_VanP = PC_point_to_CC(PC_NormVP, Normalization, height, width);
    // printf("CC_VanP: %f, %f\n", CC_VanP.x, CC_VanP.y);

    return CC_VanP;
}

#endif /*RASTERSPACE_H*/