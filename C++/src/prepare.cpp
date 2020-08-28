//
// Created by Zilin Xiao on 2020/6/5.
//

#include "prepare.h"
#include "camcal.h"
#include "drawROI.h"
#include "../include/rapidjson/document.h"
#include "exception.h"
//! Some steps before detection:
//! 0. Configure config.json file.
//! 1. Camera Calibration
//! 2. Draw ROI
//! 3. Draw Stop line & Car Lane
//! 4. Select Traffic Light Region

int main(int agrc, char** argv){
    char buffer[8192] = {0}, ch;
    int i = 0;
    FILE *f = fopen(argv[1], "r");
    while(EOF != (ch = fgetc(f))){
        buffer[i++] = ch;
    }
    fclose(f);
    rapidjson::Document d;
    d.Parse(buffer);
    auto doc = d["Preparation"].GetObject();

    cv::VideoCapture cap(doc["vdo_path"].GetString());
    if(!cap.isOpened()){
        throw FileNotFound(doc["vdo_path"].GetString());
    }
    cv::Mat frame;
    cap.read(frame); // get the first frame

    CamCalibrator camCalibrator(argv[1], frame); // 1. Camera Calibration
    drawROI(frame, doc["mask_path"].GetString()); // 2. Draw ROI

    // 3. Draw Stop line & Car Lane
    // 4. Select Traffic Light Region
    drawNotation(frame, doc["out_annotation"].GetString());
}