//
// Created by godkillerxiao on 2020/5/1.
//

#ifndef DARKNET_PLATERECOGNITION_H
#define DARKNET_PLATERECOGNITION_H

#include <rapidjson/document.h>
#include <opencv2/core/types.hpp>
#include <lpr/Pipeline.h>
#include <algorithm>
namespace pr{
    class PlateWrapper final{
    public:
        explicit PlateWrapper(char* buffer);
        ~PlateWrapper();
        void DetectMat(const cv::Mat& img);
        const std::vector<pr::PlateInfo> &getRes() const;

    private:
        //! Pipeline
        pr::PipelinePR* prc;
        //! Recognition Results
        std::vector<pr::PlateInfo> res;
        //! Threshold of Confidence
        float conf_thld;
    };
}

cv::Mat ImageCropPadding(const cv::Mat& srcImage, const cv::Rect& rect);

#endif //DARKNET_PLATERECOGNITION_H
