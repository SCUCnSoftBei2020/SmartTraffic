//
// Created by Zilin Xiao on 2020/6/5.
//

#ifndef DARKNET_DRAWROI_H
#define DARKNET_DRAWROI_H

#include <opencv2/core/mat.hpp>

void drawROI(const cv::Mat& src, const char* out_mask_path);
void drawNotation(const cv::Mat& src, const char* out_not_path);

#endif //DARKNET_DRAWROI_H
