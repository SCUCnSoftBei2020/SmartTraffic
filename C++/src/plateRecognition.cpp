//
// Created by godkillerxiao on 2020/5/1.
//

#include "plateRecognition.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat ImageCropPadding(const cv::Mat& srcImage, const cv::Rect& rect)
{
    //cv::Mat srcImage = image.clone();
    int crop_x1 = cv::max(0, rect.x);
    int crop_y1 = cv::max(0, rect.y);
    int crop_x2 = cv::min(srcImage.cols, rect.x + rect.width); // 图像范围 0到cols-1, 0到rows-1
    int crop_y2 = cv::min(srcImage.rows, rect.y + rect.height);


    int left_x = (-rect.x);
    int top_y = (-rect.y);
    int right_x = rect.x + rect.width - srcImage.cols;
    int down_y = rect.y + rect.height - srcImage.rows;
    //cv::Mat roiImage = srcImage(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));
    cv::Mat roiImage = srcImage(cv::Rect(crop_x1, crop_y1, (crop_x2 - crop_x1), (crop_y2 - crop_y1)));


    if (top_y > 0 || down_y > 0 || left_x > 0 || right_x > 0)//只要存在边界越界的情况，就需要边界填充
    {
        left_x = (left_x > 0 ? left_x : 0);
        right_x = (right_x > 0 ? right_x : 0);
        top_y = (top_y > 0 ? top_y : 0);
        down_y = (down_y > 0 ? down_y : 0);
        //cv::Scalar(0,0,255)指定颜色填充
        cv::copyMakeBorder(roiImage, roiImage, top_y, down_y, left_x, right_x, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 255));
        //cv::copyMakeBorder(roiImage, roiImage, top_y, down_y, left_x, right_x, cv::BORDER_REPLICATE);//复制最边缘像素
        //cv::copyMakeBorder(roiImage, roiImage, top_y, down_y, left_x, right_x, BORDER_REFLECT_101);  //边缘对称法填充
    }
    //else//若不存在边界越界的情况，则不需要填充了
    //{
    //  destImage = roiImage;
    //}
    return roiImage;
}



const std::vector<pr::PlateInfo> &pr::PlateWrapper::getRes() const {
    return res;
}

pr::PlateWrapper::~PlateWrapper() {
    delete prc;
}

pr::PlateWrapper::PlateWrapper(char *buffer) {
    rapidjson::Document d;
    d.Parse(buffer);
    auto doc = d["Recognition"].GetObject();
    std::string paths[9];
    int i = 0;
    for(auto& v : doc["plate_model_paths"].GetArray()){
        paths[i++] = v.GetString();
        if(i >= 9) break;
    }

    prc = new pr::PipelinePR(paths[0], paths[1], paths[2], paths[3],
            paths[4], paths[5], paths[6], paths[7], paths[8]);
}

void pr::PlateWrapper::DetectMat(const cv::Mat &img) {
    res = prc->RunPiplineAsImage(img, pr::SEGMENTATION_FREE_METHOD);
    res.erase(std::remove_if(res.begin(), res.end(), [=](const pr::PlateInfo& p){return p.confidence < conf_thld;}), res.end());
}
