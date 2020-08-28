//
// Created by godkillerxiao on 2020/5/21.
//

#ifndef DARKNET_SPEEDESTIMATION_H
#define DARKNET_SPEEDESTIMATION_H
#define RAPIDJSON_HAS_STDSTRING 1

#include <sys/stat.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <lpr/PlateInfo.h>
#include <darknet.h>

class TrackNode{
public:
    TrackNode() = default;
    TrackNode(int _frm_id, int _obj_id, const cv::Rect& _bbox,
            float _detConf, const cv::Point2f& _point2d,
            const cv::Point3f& _point3d, char* _classtype,
            float _depth);
    ~TrackNode() = default;

    void setFrmId(int frmId);

    void setObjId(int objId);

    void setBbox(const cv::Rect &bbox);

    void setDetConf(float detConf);

    void setPoint2D(const cv::Point2f &point2D);

    void setPoint3D(const cv::Point3f &point3D);

    void setDepth(float depth);

    void setSpd(float spd);

    int getFrmId() const;

    int getObjId() const;

    const cv::Rect &getBbox() const;

    float getDetConf() const;

    const cv::Point2f &getPoint2D() const;

    const cv::Point3f &getPoint3D() const;

    const char *getClasstype() const;

    float getDepth() const;

    float getSpd() const;

    static cv::Point3f pro2dto3d(const cv::Point2f& point2d, const float proMat[12], int nLenUnit = 1000);

private:
    int frm_id{};
    int obj_id{};
    cv::Rect bbox;
    float detConf{};
    cv::Point2f point2d;
    cv::Point3f point3d;
    char classtype[16]{};
    float depth{};
    float spd{};
};

class OtherNode final{
public:
    OtherNode(std::string className, float detConf, const cv::Rect &bbox);
    const std::string &getClassName() const;

    void setClassName(const std::string &className);

    float getDetConf() const;

    void setDetConf(float detConf);

    const cv::Rect &getBbox() const;

    void setBbox(const cv::Rect &bbox);

    const cv::Point2f &getPoint() const;

private:
    std::string class_name;
    float det_conf;
    cv::Rect bbox;
    cv::Point2f point;
};

class Vehicle final{
public:
    std::string plateInfo;
    float plateConf = 0.0;
    cv::Rect plateBBox;
    cv::Rect carBBox;
    std::string carType;
    float typeConf = 0.0;
    std::vector<float> vAvgSpd;
};

class Settings{
public:
    Settings() = default;
    explicit Settings(char* json_path);
    ~Settings() = default;
    int frame_gap; // in frame counts
    float speed_scale;
    float spd_std_thld;
    float spd_low_bound;
    float spd_stop_thld;
    float det_thld;
    float frm_rate;
    int traj_length;  // in frames
    cv::Size frm_size;
    bool output_vdo_flag;
    bool show_when_computing;
    int frame_thld;
    char out_vdo_path[256]{};
    char cam_cal_path[256]{};
    char in_trk_path[256]{};
    char in_vdo_path[256]{};
    char out_img_path[256]{};
    char not_path[256]{};
    char traffic_path[256]{};
    char buffer[8192]{};
};

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif //DARKNET_SPEEDESTIMATION_H
