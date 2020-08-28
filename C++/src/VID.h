#pragma once

#ifndef __MOBILENET_DNN_H__
#define __MOBILENET_DNN_H__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <ctime>
#include <algorithm>
#include <map>
#include <iostream>


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;

using namespace std;

class VID{
public:
    VID() = default;
    explicit VID(const string& model_path);
    explicit VID(char* buffer);
    ~VID() = default;
    pair<string, float> recognize(const Mat& img);
    bool enabled;

private:
    Net classNet;
    std::vector<std::string> car_types;
};

#endif //__MOBILENET_DNN_H__
