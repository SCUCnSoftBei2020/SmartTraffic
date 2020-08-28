//
// Created by Zilin Xiao on 2020/5/25.
//

#include "camcal.h"
#include "../include/rapidjson/document.h"

VanLnsSelector CamCalibrator::vanLnsSelector = VanLnsSelector(); // init static member to avoid exception

camcal::Settings::Settings(char *json_path) {
    char buffer[8192] = {0}, ch;
    int i = 0;
    FILE *f = fopen(json_path, "r");
    while(EOF != (ch = fgetc(f))){
        buffer[i++] = ch;
    }
    fclose(f);
    rapidjson::Document d;
    d.Parse(buffer);
    auto doc = d["Camera_Calibration"].GetObject();
    // memcpy(in_frm_path, doc["in_frm_path"].GetString(), 256);
    memcpy(out_cam_path, doc["out_cam_path"].GetString(), 256);
    draw_line_flag = doc["draw_line_flag"].GetBool();
    len_unit = doc["len_unit"].GetInt();
    if(!draw_line_flag){
        v_point_l = cv::Point2f(doc["v_point_l"].GetArray()[0].GetInt(),
                                doc["v_point_l"].GetArray()[1].GetInt());
        v_point_r = cv::Point2f(doc["v_point_r"].GetArray()[0].GetInt(),
                                doc["v_point_r"].GetArray()[1].GetInt());
    }
    cam_height = doc["cam_height"].GetFloat();
    grid_den_l = doc["grid_den_l"].GetFloat();
    grid_den_r = doc["grid_den_r"].GetFloat();
}

CamCalibrator::CamCalibrator(char *json_path, const cv::Mat& src) {
    ori_img = src.clone();
    s = camcal::Settings(json_path);
    std::vector<cv::Point> vVanPts = select_van_lines();
    v_point_r = vVanPts[0];
    v_point_l = vVanPts[1];
    img_size = cv::Size(ori_img.cols, ori_img.rows);
    
    assert(vVanPts.size() == 2);
    // set center
    prin_point.x = (float)ori_img.cols / 2;
    prin_point.y = (float)ori_img.rows / 2;

    while(compCamParams()){
        printf("Selected lines have no solution for Vanishing Points!\n");
        vanLnsSelector.selected_nds.clear();
        vVanPts = select_van_lines();
    };  // loop until not NaN
    
    cv::Point stGridPt = calcStGrdPt();
    pltsave(stGridPt);
}

std::vector<cv::Point> CamCalibrator::select_van_lines() {
    std::vector<cv::Point> vVanPts;
    if (s.draw_line_flag){
        printf("Vanishing Line Selector: (Could be moved to frontend)\n");
        vanLnsSelector.mod_img = ori_img.clone();
        cv::namedWindow("Selector of vanishing lines", CV_WINDOW_NORMAL);
        cv::imshow("Selector of vanishing lines", vanLnsSelector.mod_img);
        cv::setMouseCallback("Selector of vanishing lines", on_mouse);
        while (true){
            int nKey = cv::waitKey(0);
            if(nKey == 27) break;
            if(nKey == 'r'){  // clear nodes vector
                vanLnsSelector.selected_nds.clear();
                vanLnsSelector.finished_sel_flag = false;
                vanLnsSelector.mod_img = ori_img.clone();
                cv::imshow("Selector of vanishing lines", vanLnsSelector.mod_img);
            }
            if(nKey == 'o' && vanLnsSelector.finished_sel_flag){
                vVanPts = compVanPts();
                break;
            }
        }
        cv::destroyWindow("Selector of vanishing lines");
    }else{
        vVanPts.push_back(s.v_point_r);
        vVanPts.push_back(s.v_point_l);
    }


    printf("Vanishing Point Right: %d, %d\n", vVanPts[0].x, vVanPts[0].y);
    printf("Vanishing Point Left: %d, %d\n", vVanPts[1].x, vVanPts[1].y);

    return vVanPts;
}

void CamCalibrator::on_mouse(int event, int x, int y, int flags, void*) {
    if(event == CV_EVENT_FLAG_LBUTTON) vanLnsSelector.addNode(x, y);
}

std::vector<cv::Point> CamCalibrator::compVanPts() {
    //! Compute Vanishing Points Based on Vanishing Lines
    if(vanLnsSelector.selected_nds.size() != 8 || !vanLnsSelector.finished_sel_flag){
        printf("Incomplete nodes list.\n");
        cv::waitKey(0);
    }
    std::vector<cv::Point> vVanPts;

    // using two lines perpendicular to each other to compute vanishing point
    double p1k1 = (double)(vanLnsSelector.selected_nds[0].y - vanLnsSelector.selected_nds[1].y) /
            (double)(vanLnsSelector.selected_nds[0].x - vanLnsSelector.selected_nds[1].x);
    double p1k2 = (double)(vanLnsSelector.selected_nds[2].y - vanLnsSelector.selected_nds[3].y) /
                  (double)(vanLnsSelector.selected_nds[2].x - vanLnsSelector.selected_nds[3].x);
    cv::Point vPt1;
    vPt1.x = ((p1k1 * vanLnsSelector.selected_nds[0].x) - (p1k2 * vanLnsSelector.selected_nds[2].x) +
            vanLnsSelector.selected_nds[2].y - vanLnsSelector.selected_nds[0].y) / (p1k1 - p1k2);
    vPt1.y = vanLnsSelector.selected_nds[0].y + ((vPt1.x - vanLnsSelector.selected_nds[0].x) * p1k1);


    double p2k1 = (double)(vanLnsSelector.selected_nds[4].y - vanLnsSelector.selected_nds[5].y) /
                  (double)(vanLnsSelector.selected_nds[4].x - vanLnsSelector.selected_nds[5].x);
    double p2k2 = (double)(vanLnsSelector.selected_nds[6].y - vanLnsSelector.selected_nds[7].y) /
                  (double)(vanLnsSelector.selected_nds[6].x - vanLnsSelector.selected_nds[7].x);
    cv::Point vPt2;

    vPt2.x = ((p2k1 * vanLnsSelector.selected_nds[4].x) - (p2k2 * vanLnsSelector.selected_nds[6].x) +
              vanLnsSelector.selected_nds[6].y - vanLnsSelector.selected_nds[4].y) / (p2k1 - p2k2);
    vPt2.y = vanLnsSelector.selected_nds[4].y + ((vPt2.x - vanLnsSelector.selected_nds[4].x) * p2k1);

    if(vPt1.x >= vPt2.x){ // the point with larger x is vPtR
        vVanPts.push_back(vPt1);
        vVanPts.push_back(vPt2);
    }else{
        vVanPts.push_back(vPt2);
        vVanPts.push_back(vPt1);
    }

    return vVanPts;
}

bool CamCalibrator::compCamParams() {
    cv::Point2f VrC, VlC, VrCRot, VlCRot;
    // intristic params of camera
    float fF, fRoll, fPitch, fYaw;
    
    VrC.x = v_point_r.x - prin_point.x;
    VrC.y = prin_point.y - v_point_r.y;
    
    VlC.x = v_point_l.x - prin_point.x;
    VlC.y = prin_point.y - v_point_l.y;
    
    // compute fRoll
    fRoll = std::atan2((VrC.y - VlC.y), (VrC.x - VlC.x));
    fRoll = (fRoll > (CV_PI / 2.0)) ? (fRoll - CV_PI) : fRoll;
    fRoll = (fRoll < (-CV_PI / 2.0)) ? (fRoll + CV_PI) : fRoll;

    VrCRot = rotPt(VrC, -fRoll);
    VlCRot = rotPt(VlC, -fRoll);
    if(-((VrCRot.y * VrCRot.y) + (VrCRot.x * VlCRot.x)) < 0){
        return true;
    }

    fF = std::sqrt(-((VrCRot.y * VrCRot.y) + (VrCRot.x * VlCRot.x)));

    fPitch = -std::atan2(VrCRot.y, fF);
    fYaw = -std::atan2((VrCRot.x * std::cos(fPitch)), fF);


    camParam.Fx = fF;
    camParam.Fy = fF;
    camParam.Cx = prin_point.x;
    camParam.Cy = prin_point.y;
    camParam.Roll = fRoll;
    camParam.Pitch = fPitch;
    camParam.Yaw = fYaw;
    camParam.Tx = 0.0f;
    
    camParam.Ty = 0.0f;
    camParam.Tz = s.cam_height * s.len_unit;

    camParam.setInParamMat(fF, fF, prin_point.x, prin_point.y);
    camParam.setRotMat(fRoll, fPitch, fYaw);
    camParam.setTntMat(0.0f, 0.0f, (s.cam_height * s.len_unit));
    camParam.calcProjMat();
    return false;
    
}

cv::Point CamCalibrator::calcStGrdPt() {
    // look for the starting grid location that will make all the grid points within the frame image
    bool bStGrdPtFlg = false;
    int nMaxDist, nMaxSumSqDist = 0;
    std::vector<cv::Point> voPt;
    cv::Point oStGrdPt;

    // iterate from smallest distance to largest distance to the original point
    while (true)
    {
        std::vector<cv::Point>().swap(voPt);
        nMaxDist = std::sqrt(nMaxSumSqDist);

        for (int i = 0; i <= nMaxDist; i++)
        {
            for (int j = i; j <= nMaxDist; j++)
            {
                if (nMaxSumSqDist == (i * i) + (j * j))
                    voPt.push_back(cv::Point(i, j));
            }
        }

        if (voPt.size())
        {
            for (int i = 0; i < voPt.size(); i++)
            {
                oStGrdPt.x = voPt[i].x; oStGrdPt.y = voPt[i].y;
                if (tstStGrdPt(oStGrdPt, camParam)) { bStGrdPtFlg = true; break; }

                if (0 < voPt[i].x)
                {
                    oStGrdPt.x = -voPt[i].x; oStGrdPt.y = voPt[i].y;
                    if (tstStGrdPt(oStGrdPt, camParam)) { bStGrdPtFlg = true; break; }
                }

                if (0 < voPt[i].y)
                {
                    oStGrdPt.x = voPt[i].x; oStGrdPt.y = -voPt[i].y;
                    if (tstStGrdPt(oStGrdPt, camParam)) { bStGrdPtFlg = true; break; }
                }

                if (0 < voPt[i].x)
                {
                    oStGrdPt.x = -voPt[i].x; oStGrdPt.y = -voPt[i].y;
                    if (tstStGrdPt(oStGrdPt, camParam)) { bStGrdPtFlg = true; break; }
                }

                if (voPt[i].x < voPt[i].y)
                {
                    oStGrdPt.x = voPt[i].y; oStGrdPt.y = voPt[i].x;
                    if (tstStGrdPt(oStGrdPt, camParam)) { bStGrdPtFlg = true; break; }
                }

                if (voPt[i].x < voPt[i].y)
                {
                    oStGrdPt.x = -voPt[i].y; oStGrdPt.y = voPt[i].x;
                    if (tstStGrdPt(oStGrdPt, camParam)) { bStGrdPtFlg = true; break; }
                }

                if ((voPt[i].x < voPt[i].y) && (0 < voPt[i].x))
                {
                    oStGrdPt.x = voPt[i].y; oStGrdPt.y = -voPt[i].x;
                    if (tstStGrdPt(oStGrdPt, camParam)) { bStGrdPtFlg = true; break; }
                }

                if ((voPt[i].x < voPt[i].y) && (0 < voPt[i].x))
                {
                    oStGrdPt.x = -voPt[i].y; oStGrdPt.y = -voPt[i].x;
                    if (tstStGrdPt(oStGrdPt, camParam)) { bStGrdPtFlg = true; break; }
                }
            }
        }

        if (bStGrdPtFlg)
            break;

        nMaxSumSqDist++;
    }

    return oStGrdPt;
}

bool CamCalibrator::tstStGrdPt(cv::Point oStGrdPt, CamParam &poCamParam) {
    int nLftEdgX = (-(IMG_EXPN_RAT - 1.0f) / 2.0f) * (float)ori_img.cols;
    int nTopEdgY = (-(IMG_EXPN_RAT - 1.0f) / 2.0f) * (float)ori_img.rows;
    int nRgtEdgX = ((IMG_EXPN_RAT + 1.0f) / 2.0f) * (float)ori_img.cols;
    int nBtmEdgY = ((IMG_EXPN_RAT + 1.0f) / 2.0f) * (float)ori_img.rows;

    cv::Point2f o2dStPt(-1.0f, -1.0f), o2dRNdPt(-1.0f, -1.0f), o2dLNdPt(-1.0f, -1.0f), o2dNdPt(-1.0f, -1.0f);

    cv::Point oNdGrdPt;
    oNdGrdPt.x = oStGrdPt.x + s.grid_den_r;
    oNdGrdPt.y = oStGrdPt.y + s.grid_den_l;

    o2dStPt = proj3d22d(cv::Point3f(oStGrdPt.x, oStGrdPt.y, 0.0f), camParam.fP, s.len_unit);

    o2dRNdPt = proj3d22d(cv::Point3f(oNdGrdPt.x, oStGrdPt.y, 0.0f), camParam.fP, s.len_unit);

    o2dLNdPt = proj3d22d(cv::Point3f(oStGrdPt.x, oNdGrdPt.y, 0.0f), camParam.fP, s.len_unit);

    o2dNdPt = proj3d22d(cv::Point3f(oNdGrdPt.x, oNdGrdPt.y, 0.0f), camParam.fP, s.len_unit);

    return ((o2dStPt.x >= nLftEdgX) && (o2dStPt.y >= nTopEdgY) && (o2dStPt.x < nRgtEdgX) && (o2dStPt.y < nBtmEdgY)) &&
           ((o2dRNdPt.x >= nLftEdgX) && (o2dRNdPt.y >= nTopEdgY) && (o2dRNdPt.x < nRgtEdgX) &&
            (o2dRNdPt.y < nBtmEdgY)) &&
           ((o2dLNdPt.x >= nLftEdgX) && (o2dLNdPt.y >= nTopEdgY) && (o2dLNdPt.x < nRgtEdgX) &&
            (o2dLNdPt.y < nBtmEdgY)) &&
           ((o2dNdPt.x >= nLftEdgX) && (o2dNdPt.y >= nTopEdgY) && (o2dNdPt.x < nRgtEdgX) && (o2dNdPt.y < nBtmEdgY));
}

void CamCalibrator::pltsave(cv::Point oStGrdPt) {
    cv::Point2f oVrExpn = projPtOrig2Expn(v_point_r, IMG_EXPN_RAT, img_size),
            oVlExpn = projPtOrig2Expn(v_point_l, IMG_EXPN_RAT, img_size);

    cv::Mat oImgExpn = genExpnImg(IMG_EXPN_RAT, img_size);
    cv::Mat oImgCent(oImgExpn,
                     cv::Rect(((IMG_EXPN_RAT - 1.0f) * img_size.width / 2.0f),
                              ((IMG_EXPN_RAT - 1.0f) * img_size.height / 2.0f),
                              img_size.width, img_size.height));
    ori_img.copyTo(oImgCent);

    FILE* pfCamParam;
    pfCamParam = std::fopen(s.out_cam_path, "w");
    float *afK, *afR, *afT, *afP;
    afK = camParam.fK;
    afR = camParam.fR;
    afT = camParam.fT;
    afP = camParam.fP;

//    std::fprintf(pfCamParam, "%.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f\n",
//                 afK[0], afK[1], afK[2], afK[3], afK[4], afK[5], afK[6], afK[7], afK[8]);
//    std::fprintf(pfCamParam, "%.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f\n",
//                 afR[0], afR[1], afR[2], afR[3], afR[4], afR[5], afR[6], afR[7], afR[8]);
//    std::fprintf(pfCamParam, "%.7f %.7f %.7f\n", afT[0], afT[1], afT[2]);
    std::fprintf(pfCamParam, "%.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f\n",
                 afP[0], afP[1], afP[2], afP[3], afP[4], afP[5], afP[6], afP[7], afP[8], afP[9], afP[10], afP[11]);
    std::fclose(pfCamParam);

    cv::Point2f o2dMeasPt;
    cv::circle(oImgExpn, oVrExpn, 3, cv::Scalar(255, 128, 0, 0), 2);
    cv::circle(oImgExpn, oVlExpn, 3, cv::Scalar(255, 128, 0, 0), 2);

    cv::Point oNdGrdPt;
    oNdGrdPt.x = oStGrdPt.x + s.grid_den_r;
    oNdGrdPt.y = oStGrdPt.y + s.grid_den_l;

//    std::vector<cv::Point> voMeasLnSegNdPt = m_oCfg.getCalMeasLnSegNdPt();
//    cv::Point oSt2dPt, oNd2dPt;
//
//    for (int i = 0; i < (voMeasLnSegNdPt.size() / 2); i++)
//    {
//        oSt2dPt = voMeasLnSegNdPt[i * 2];
//        oNd2dPt = voMeasLnSegNdPt[i * 2 + 1];
//        cv::line(oImgExpn, projPtOrig2Expn(oSt2dPt, IMG_EXPN_RAT, m_oCfg.getFrmSz()),
//                 projPtOrig2Expn(oNd2dPt, IMG_EXPN_RAT, m_oCfg.getFrmSz()), cv::Scalar(0, 255, 0, 0), 3);
//    }

    // draw points
    for (int iL = oStGrdPt.y; iL < oNdGrdPt.y; iL++)
    {
        for (int iR = oStGrdPt.x; iR < oNdGrdPt.x; iR++)
        {
            o2dMeasPt = proj3d22d(cv::Point3f(iR, iL, 0.0f), camParam.fP, s.len_unit);
            o2dMeasPt = projPtOrig2Expn(o2dMeasPt, IMG_EXPN_RAT, img_size);

            if ((0 <= o2dMeasPt.x) && (oImgExpn.size().width > o2dMeasPt.x) &&
                (0 <= o2dMeasPt.y) && (oImgExpn.size().height > o2dMeasPt.y))
                cv::circle(oImgExpn, o2dMeasPt, 3, cv::Scalar(0, 0, 255, 0), 10);
        }
    }

    cv::Mat oImgDisp = cv::Mat(cv::Size(1920, (oImgExpn.size().height * 1920 / oImgExpn.size().width)), CV_8UC3);
    cv::resize(oImgExpn, oImgDisp, oImgDisp.size());
    cv::namedWindow("3D grid on ground plane", CV_WINDOW_NORMAL);
    cv::imshow("3D grid on ground plane", oImgDisp);
    cv::waitKey(0);

}


void VanLnsSelector::addNode(int x, int y) {
    cv::Point curr_nd(x, y);
    if(selected_nds.size() % 2 == 1){  // ready to draw line
        selected_nds.emplace_back(curr_nd);
        cv::Point prev_nd = selected_nds[selected_nds.size() - 2];
        cv::circle(mod_img, curr_nd, 6, cv::Scalar(255, 0, 0), 1, CV_AA);
        cv::line(mod_img, prev_nd, curr_nd, cv::Scalar(255,255,255), 2, CV_AA);
        cv::imshow("Selector of vanishing lines", mod_img);
        if(selected_nds.size() == 2 || selected_nds.size() == 6){
            printf("Select another vanishing line parallel to the previous one.\n");
        }
        else if (selected_nds.size() == 4){
            printf("Select another vanishing line perpendicular to the previous pair of lines.\n");
        }else{
            finished_sel_flag = true;
        }

    }else{
        selected_nds.emplace_back(curr_nd);
        cv::circle(mod_img, curr_nd, 6, cv::Scalar(255, 0, 0), 1, CV_AA);
        cv::imshow("Selector of vanishing lines", mod_img);
    }
}

void CamParam::setInParamMat(float fFx, float fFy, float fCx, float fCy) {
    fK[0] = fFx;
    fK[1] = 0.0f;
    fK[2] = fCx;
    fK[3] = 0.0f;
    fK[4] = fFy;
    fK[5] = fCy;
    fK[6] = 0.0f;
    fK[7] = 0.0f;
    fK[8] = 1.0f;
}

void CamParam::setRotMat(float fRoll, float fPitch, float fYaw) {
    fR[0] = (-std::cos(fRoll) * std::sin(fYaw)) - (std::sin(fRoll) * std::sin(fPitch) * std::cos(fYaw));
    fR[1] = (-std::cos(fRoll) * std::cos(fYaw)) - (std::sin(fRoll) * std::sin(fPitch) * std::cos(fYaw));
    fR[2] = std::sin(fRoll) * std::cos(fPitch);
    fR[3] = (-std::sin(fRoll) * std::sin(fYaw)) + (std::cos(fRoll) * std::sin(fPitch) * std::cos(fYaw));
    fR[4] = (-std::sin(fRoll) * std::cos(fYaw)) - (std::cos(fRoll) * std::sin(fPitch) * std::sin(fYaw));
    fR[5] = -std::cos(fRoll) * std::cos(fPitch);
    fR[6] = std::cos(fPitch) * std::cos(fYaw);
    fR[7] = -std::cos(fPitch) * std::sin(fYaw);
    fR[8] = std::sin(fPitch);
}

void CamParam::setTntMat(float fTx, float fTy, float fTz) {
    fT[0] = fTx;
    fT[1] = fTy;
    fT[2] = fTz;
}

void CamParam::calcProjMat() {
    cv::Mat oMatK(3, 3, CV_32FC1, fK);
    cv::Mat oMatR(3, 3, CV_32FC1, fR);

    // T = -Rt
    cv::Mat oMatT(3, 1, CV_32FC1);
    oMatT.ptr<float>(0)[0] = -fT[0];
    oMatT.ptr<float>(1)[0] = -fT[1];
    oMatT.ptr<float>(2)[0] = -fT[2];
    oMatT = oMatR * oMatT;

    // P = K [R|T]
    cv::Mat oMatP(3, 4, CV_32FC1);
    oMatP.ptr<float>(0)[0] = fR[0];
    oMatP.ptr<float>(0)[1] = fR[1];
    oMatP.ptr<float>(0)[2] = fR[2];
    oMatP.ptr<float>(1)[0] = fR[3];
    oMatP.ptr<float>(1)[1] = fR[4];
    oMatP.ptr<float>(1)[2] = fR[5];
    oMatP.ptr<float>(2)[0] = fR[6];
    oMatP.ptr<float>(2)[1] = fR[7];
    oMatP.ptr<float>(2)[2] = fR[8];
    oMatP.ptr<float>(0)[3] = oMatT.ptr<float>(0)[0];
    oMatP.ptr<float>(1)[3] = oMatT.ptr<float>(1)[0];
    oMatP.ptr<float>(2)[3] = oMatT.ptr<float>(2)[0];
    oMatP = oMatK * oMatP;

    fP[0] = oMatP.ptr<float>(0)[0];
    fP[1] = oMatP.ptr<float>(0)[1];
    fP[2] = oMatP.ptr<float>(0)[2];
    fP[3] = oMatP.ptr<float>(0)[3];
    fP[4] = oMatP.ptr<float>(1)[0];
    fP[5] = oMatP.ptr<float>(1)[1];
    fP[6] = oMatP.ptr<float>(1)[2];
    fP[7] = oMatP.ptr<float>(1)[3];
    fP[8] = oMatP.ptr<float>(2)[0];
    fP[9] = oMatP.ptr<float>(2)[1];
    fP[10] = oMatP.ptr<float>(2)[2];
    fP[11] = oMatP.ptr<float>(2)[3];
}
