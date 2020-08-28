//
// Created by Zilin Xiao on 2020/5/25.
//
#pragma once
#ifndef DARKNET_CAMCAL_H
#define DARKNET_CAMCAL_H
#define IMG_EXPN_RAT (2.0f)

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

template <typename T> T deg2rad(T deg) { return deg * (CV_PI / 180.0); }

namespace camcal{
    class Settings{
    public:
        Settings() = default;
        explicit Settings(char* json_path);
        char in_frm_path[256]{};
        char out_cam_path[256]{};
        bool draw_line_flag{};
        int len_unit{};
        cv::Point2f v_point_l, v_point_r;
        float cam_height{};
        float grid_den_l{}, grid_den_r{};
    };
}

class VanLnsSelector{
public:
    VanLnsSelector() = default;
    ~VanLnsSelector() = default;

    cv::Mat mod_img;
    std::vector<cv::Point> selected_nds;
    bool finished_sel_flag{};
    void addNode(int x, int y);
};


class CamParam
{
public:
    struct SParamRng
    {
        // X-Z ground plane
        // s m' = K [R|t] M' (P = K [R|t])
        //   [u]   [fx 0  cx] [r11 r12 r13] [1 0 0 t1] [X]
        // s [v] = [0  fy cy] [r21 r22 r23] [0 1 0 t2] [Y]
        //   [1]   [0  0  1 ] [r31 r32 r33] [0 0 1 t3] [Z]
        //                                             [1]
        // skew = 0
        // X&Z on the ground, Y straight downwards
        // (0) originally CCS parallel with WCS
        // (1) translate upwards by t
        // (2) rotate yaw(pan) degrees around Y axis
        // (3) rotate pitch(tilt) degrees around X axis
        // (4) rotate roll degrees around Z axis
        // R = RZ * RX * RY
        //      [cos(roll) -sin(roll) 0          ]
        // RZ = [sin(roll) cos(roll)  0          ]
        //      [0         0          1          ]
        //      [1         0          0          ]
        // RX = [0         cos(pitch) -sin(pitch)]
        //      [0         sin(pitch) cos(pitch) ]
        //      [cos(yaw)  0          sin(yaw)   ]
        // RY = [0         1          0          ]
        //      [-sin(yaw) 0          cos(yaw)   ]
        // r11 = cos(roll)cos(yaw) - sin(roll)sin(pitch)sin(yaw)
        // r12 = -sin(roll)cos(pitch)
        // r13 = cos(roll)sin(yaw) + sin(roll)sin(pitch)cos(yaw)
        // r21 = sin(roll)cos(yaw) + cos(roll)sin(pitch)sin(yaw)
        // r22 = cos(roll)cos(pitch)
        // r23 = sin(roll)sin(yaw) - cos(roll)sin(pitch)cos(yaw)
        // r31 = -cos(pitch)sin(yaw)
        // r32 = sin(pitch)
        // r33 = cos(pitch)cos(yaw)
        // t1 = tx(0)
        // t2 = ty(-Hc)
        // t3 = tz(0)

        SParamRng()
        {
            fFxMax = 5000, fFxMin = 0;
            fFyMax = 5000, fFyMin = 0;
            fCxMax = 5000, fCxMin = 0;
            fCyMax = 5000, fCyMin = 0;
            fRollMax = deg2rad(90), fRollMin = deg2rad(-90);
            fPitchMax = deg2rad(90), fPitchMin = deg2rad(-90);
            fYawMax = deg2rad(90), fYawMin = deg2rad(-90);
            fTxMax = 10, fTxMin = -10;
            fTyMax = 10, fTyMin = -10;
            fTzMax = 10, fTzMin = -10;
        }

        float fFxMax, fFxMin;	// camera focal length
        float fFyMax, fFyMin;	// camera focal length fy = fx * a (aspect ratio, close to 1)
        float fCxMax, fCxMin;	// optical center/principal point
        float fCyMax, fCyMin;	// optical center/principal point
        float fRollMax{}, fRollMin{};	// roll angle
        float fPitchMax{}, fPitchMin{};	// pitch(tilt) angle
        float fYawMax{}, fYawMin{};	// yaw(pan) angle
        float fTxMax, fTxMin;
        float fTyMax, fTyMin;
        float fTzMax, fTzMin;
    };

    CamParam() = default;
    ~CamParam() = default;

    //! sets the matrix of intrinsic parameters
    void setInParamMat(float fFx, float fFy, float fCx, float fCy);
    //! sets the matrix of rotation
    void setRotMat(float fRoll, float fPitch, float fYaw);
    //! sets the matrix of translation
    void setTntMat(float fTx, float fTy, float fTz);
    //! calculates the projective matrix
    void calcProjMat();

//    //! initializes camera model
//    void initCamMdl(SParamRng sParamRng);

    float fK[9];
    float fR[9];
    float fT[3];
    float fP[12];

    float Fx;
    float Fy;
    float Cx;
    float Cy;
    float Roll;
    float Pitch;
    float Yaw;
    float Tx;
    float Ty;
    float Tz;

    float ReprojErr;
};

class CamCalibrator{
public:
    CamCalibrator() = default;
    CamCalibrator(char* json_path, const cv::Mat& src);
    ~CamCalibrator() = default;
    std::vector<cv::Point> select_van_lines();
    camcal::Settings s;
    cv::Mat ori_img;
    cv::Size img_size;
    cv::Point2f v_point_l, v_point_r;
    CamParam camParam{};
    static void on_mouse(int event, int x, int y, int flags, void*);
    static VanLnsSelector vanLnsSelector;

private:
    cv::Point2f prin_point;
    std::vector<cv::Point> compVanPts();
    bool compCamParams();
    cv::Point calcStGrdPt();
    cv::Mat ImgBg;
    void pltsave(cv::Point oStGrdPt);

    bool tstStGrdPt(cv::Point oStGrdPt, CamParam& poCamParam);


    static cv::Point2f rotPt(const cv::Point2f& oPt, float fAng){
        return cv::Point2f(((oPt.x * std::cos(fAng)) - (oPt.y * std::sin(fAng))),
                           ((oPt.x * std::sin(fAng)) + (oPt.y * std::cos(fAng))));
    }
    static cv::Point2f proj3d22d(cv::Point3f o3dPt, float afProjMat[12], int nLenUnit = 1)
    {
        cv::Mat oMatP(3, 4, CV_32FC1, afProjMat);

        cv::Mat oMatM3d(4, 1, CV_32FC1);
        oMatM3d.at<float>(0, 0) = o3dPt.x * nLenUnit;
        oMatM3d.at<float>(1, 0) = o3dPt.y * nLenUnit;
        oMatM3d.at<float>(2, 0) = o3dPt.z * nLenUnit;
        oMatM3d.at<float>(3, 0) = 1.0f;

        cv::Mat oMatM2d(3, 1, CV_32FC1);
        oMatM2d = oMatP * oMatM3d;

        cv::Point2f o2dPt = cv::Point2f(oMatM2d.at<float>(0, 0) / oMatM2d.at<float>(2, 0),
                                        oMatM2d.at<float>(1, 0) / oMatM2d.at<float>(2, 0));

        return o2dPt;
    }

    static cv::Point2f projPtOrig2Expn(cv::Point2f oPtOrig, float fExpnRat, cv::Size oFrmSz)
    {
        return cv::Point2f(oPtOrig.x + (oFrmSz.width * (fExpnRat - 1.0f) / 2.0f),
                           oPtOrig.y + (oFrmSz.height * (fExpnRat - 1.0f) / 2.0f));
    }

    static cv::Mat genExpnImg(float fExpnRat, cv::Size oFrmSz)
    {
        cv::Mat oImgExpn;

        if (1.0f >= fExpnRat)
        {
            oImgExpn = cv::Mat::zeros(cv::Size(oFrmSz.width, oFrmSz.height), CV_8UC3);
            return oImgExpn;
        }
        else
        {
            oImgExpn = cv::Mat::zeros(cv::Size((oFrmSz.width * fExpnRat), (oFrmSz.height * fExpnRat)), CV_8UC3);
            return oImgExpn;
        }
    }
};

 // init static member to avoid exception


#endif //DARKNET_CAMCAL_H
