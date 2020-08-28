//
// Created by godkillerxiao on 2020/5/21.
//

#include "speedEstimation.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "exception.h"
#include "plateRecognition.h"
#include "VID.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/freetype.hpp>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <regex>
#include <utility>
#define MIN(A,B) ((A) <= (B) ? (A) : (B)) 
#define EPSILON 0.000001
#define MAX_FRAME 6000
using std::vector;

short flow_count[MAX_FRAME], obj_set[999];

struct carinfo{
    short obj_id;
    short frame_id;
    cv::Rect bbox;
};
carinfo vzeroBuffer[99999], speedingBuffer[99999];  // buffers of spding&parking cars's info
int     vzero_cnt, spding_cnt;

struct LINE{
    int k,b;
};
LINE  stop_line;      //  y=k*x+b

short is_red[MAX_FRAME][4];
int   zone[999];
int   lane_num,SidewalkNum,npark_num,stopline,car_light,npark_v;
vector<Point> ctrs[10],redlight[5],lane[5];
vector<vector<Point> > contours;

static vector<vector<OtherNode> > otherNodesList; // otherNodesList[i][j] the j-th object appears at frame i
static vector<vector<TrackNode> > trkNodesList; // trkNodesList[i]: all frames where obj_i appears

static int getUtf8LetterNumber(const char *s) {
    int i = 0, j = 0;
    while (s[i]) {
        if ((s[i] & 0xc0) != 0x80) j++;
        i++;
    }
    return j;
}

static regex plateRegex("([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]{3}|[A-Z]{1})[A-Z]{1}[A-HJ-NP-Z0-9]{4}([A-HJ-NP-Z0-9]{1}|[挂学警港澳]{3})");
static string plateFilter(const string &plate) {
    smatch sm;
    if(regex_match(plate, sm, plateRegex)){
        return sm[0];
    }
    return "";
}

class Rounder {
public:
    int bits;
    stringstream ss;
    float ans;

    Rounder(int _bits) : bits(_bits) {}

    float GetAns(float f) {
        ss << fixed << setprecision(bits) << f;
        ss >> ans;
        return ans;
    }
};

bool cmpDepth(const TrackNode &trknd1, const TrackNode &trknd2) {
    return trknd1.getDepth() < trknd2.getDepth(); // in the ascending order
}

bool cmpFrm(const TrackNode &trknd1, const TrackNode &trknd2) {
    return trknd1.getFrmId() < trknd2.getFrmId();
}

TrackNode::TrackNode(int _frm_id, int _obj_id, const cv::Rect &_bbox, float _detConf, const cv::Point2f &_point2d,
                     const cv::Point3f &_point3d, char *_classtype, float _depth) {
    setFrmId(_frm_id);
    setObjId(_obj_id);
    setBbox(_bbox);
    setDetConf(_detConf);
    setPoint2D(_point2d);
    setPoint3D(_point3d);
    strcpy(classtype, _classtype);
    setDepth(_depth);
}

int TrackNode::getFrmId() const {
    return frm_id;
}

int TrackNode::getObjId() const {
    return obj_id;
}

const cv::Rect &TrackNode::getBbox() const {
    return bbox;
}

float TrackNode::getDetConf() const {
    return detConf;
}

const cv::Point2f &TrackNode::getPoint2D() const {
    return point2d;
}

const cv::Point3f &TrackNode::getPoint3D() const {
    return point3d;
}

const char *TrackNode::getClasstype() const {
    return classtype;
}

float TrackNode::getDepth() const {
    return depth;
}

float TrackNode::getSpd() const {
    return spd;
}

void TrackNode::setFrmId(int frmId) {
    frm_id = frmId;
}

void TrackNode::setObjId(int objId) {
    obj_id = objId;
}

void TrackNode::setBbox(const cv::Rect &bbox) {
    TrackNode::bbox = bbox;
}

void TrackNode::setDetConf(float detConf) {
    TrackNode::detConf = detConf;
}

void TrackNode::setPoint2D(const cv::Point2f &point2D) {
    point2d = point2D;
}

void TrackNode::setPoint3D(const cv::Point3f &point3D) {
    point3d = point3D;
}

void TrackNode::setDepth(float depth) {
    TrackNode::depth = depth;
}

void TrackNode::setSpd(float spd) {
    if (spd >= 0)
        TrackNode::spd = spd;
}

class SpeedSumHelper {
public:
    float operator()(float lhs, const TrackNode &rhs) {
        return (lhs + rhs.getSpd());
    }
};

cv::Point3f TrackNode::pro2dto3d(const cv::Point2f &point2d, const float *proMat, int nLenUnit) {
    cv::Point3f point3d;

    cv::Mat oMatA(3, 3, CV_64F);
    oMatA.at<double>(0, 0) = proMat[0];
    oMatA.at<double>(0, 1) = proMat[1];
    oMatA.at<double>(0, 2) = -point2d.x;
    oMatA.at<double>(1, 0) = proMat[4];
    oMatA.at<double>(1, 1) = proMat[5];
    oMatA.at<double>(1, 2) = -point2d.y;
    oMatA.at<double>(2, 0) = proMat[8];
    oMatA.at<double>(2, 1) = proMat[9];
    oMatA.at<double>(2, 2) = -1.0;

    cv::Mat oMatAInv(3, 3, CV_64F);
    cv::invert(oMatA, oMatAInv, cv::DECOMP_SVD);

    cv::Mat oMatB(3, 1, CV_64F);
    oMatB.at<double>(0, 0) = -proMat[3];
    oMatB.at<double>(1, 0) = -proMat[7];
    oMatB.at<double>(2, 0) = -proMat[11];

    cv::Mat oMatM(3, 1, CV_64F);
    oMatM = oMatAInv * oMatB;

    point3d = cv::Point3f(oMatM.at<double>(0, 0), oMatM.at<double>(1, 0), 0.0f) / nLenUnit;
    return point3d;
}


Settings::Settings(char *json_path) {
    char ch;
    int i = 0;
    FILE *f = fopen(json_path, "r");
    if(f == nullptr){
        fputs("Json Config Not Found!", stderr);
        return;
    }
    while (EOF != (ch = fgetc(f))) {
        buffer[i++] = ch;
    }
    fclose(f);
    rapidjson::Document d;
    d.Parse(buffer);
//    auto doc1 = d["Plate_Recognition"].GetObject();
//    for(auto& v : doc1["model_paths"].GetArray()){
//        printf("%s\n", v.GetString());
//    }
    auto doc = d["Detection"].GetObject();
    memcpy(traffic_path, doc["traffic_path"].GetString(), 256);

    for (auto &m : d["Speed_Estimation"].GetObject()) {
        if (strcmp(m.name.GetString(), "frame_gap") == 0) frame_gap = m.value.GetInt();
        if (strcmp(m.name.GetString(), "speed_scale") == 0) speed_scale = m.value.GetFloat();
        if (strcmp(m.name.GetString(), "speed_std_thld") == 0) spd_std_thld = m.value.GetFloat();
        if (strcmp(m.name.GetString(), "spd_low_bound") == 0) spd_low_bound = m.value.GetFloat();
        if (strcmp(m.name.GetString(), "spd_stop_thld") == 0) spd_stop_thld = m.value.GetFloat();
        if (strcmp(m.name.GetString(), "det_thld") == 0) det_thld = m.value.GetFloat();
        // if(strcmp(m.name.GetString(), "frm_rate") == 0) frm_rate = m.value.GetFloat();
        if (strcmp(m.name.GetString(), "traj_length") == 0) traj_length = m.value.GetInt();
        if (strcmp(m.name.GetString(), "output_vdo_flag") == 0) output_vdo_flag = m.value.GetBool();
        if (strcmp(m.name.GetString(), "show_when_computing") == 0) show_when_computing = m.value.GetBool();
        if (strcmp(m.name.GetString(), "frames_thld") == 0) frame_thld = m.value.GetInt();
//        if(strcmp(m.name.GetString(), "frm_size") == 0) {
//            rapidjson::Value::ConstValueIterator citr = m.value.GetArray().Begin();
//            frm_size.width = citr->GetInt();
//            citr++;
//            frm_size.height = citr->GetInt();
//        }
        // if(strcmp(m.name.GetString(), "out_spd_path") == 0) memcpy(out_spd_path, m.value.GetString(), 256);
        if (strcmp(m.name.GetString(), "out_vdo_path") == 0) memcpy(out_vdo_path, m.value.GetString(), 256);
        if (strcmp(m.name.GetString(), "cam_cal_path") == 0) memcpy(cam_cal_path, m.value.GetString(), 256);
        if (strcmp(m.name.GetString(), "in_trk_path") == 0) memcpy(in_trk_path, m.value.GetString(), 256);
        if (strcmp(m.name.GetString(), "in_vdo_path") == 0) memcpy(in_vdo_path, m.value.GetString(), 256);
        if (strcmp(m.name.GetString(), "out_img_path") == 0) memcpy(out_img_path, m.value.GetString(), 256);

    }

    for (auto &m : d["Preparation"].GetObject()) {
        if (strcmp(m.name.GetString(), "out_annotation") == 0)
        {
            memcpy(not_path, m.value.GetString(), 256);
            break;
        }
    }
}


/** #######################Judge whether Point in Polygon 2D#####################################**/
bool IsPointOnLine(double px0, double py0, double px1, double py1, double px2, double py2)
{
	bool flag = false;
	double d1 = (px1 - px0) * (py2 - py0) - (px2 - px0) * (py1 - py0);
	if ((abs(d1) < EPSILON) && ((px0 - px1) * (px0 - px2) <= 0) && ((py0 - py1) * (py0 - py2) <= 0))
	{
		flag = true;
	}
	return flag;
}

bool IsIntersect(double px1, double py1, double px2, double py2, double px3, double py3, double px4, double py4)
{
	bool flag = false;
	double d = (px2 - px1) * (py4 - py3) - (py2 - py1) * (px4 - px3);
	if (d != 0)
	{
		double r = ((py1 - py3) * (px4 - px3) - (px1 - px3) * (py4 - py3)) / d;
		double s = ((py1 - py3) * (px2 - px1) - (px1 - px3) * (py2 - py1)) / d;
		if ((r >= 0) && (r <= 1) && (s >= 0) && (s <= 1))
		{
			flag = true;
		}
	}
	return flag;
}

bool Point_In_Polygon_2D(double x, double y, const vector<Point> &POL)
{	
	bool isInside = false;
	int count = 0;
	
	double minX = DBL_MAX;
	for (int i = 0; i < POL.size(); i++)
	{
		minX = MIN(minX, POL[i].x);
	}

	double px = x;
	double py = y;
	double linePoint1x = x;
	double linePoint1y = y;
	double linePoint2x = minX -10;
	double linePoint2y = y;

	for (int i = 0; i < POL.size() - 1; i++)
	{	
		double cx1 = POL[i].x;
		double cy1 = POL[i].y;
		double cx2 = POL[i + 1].x;
		double cy2 = POL[i + 1].y;
				
		if (IsPointOnLine(px, py, cx1, cy1, cx2, cy2))
		{
			return true;
		}

		if (fabs(cy2 - cy1) < EPSILON) 
		{
			continue;
		}

		if (IsPointOnLine(cx1, cy1, linePoint1x, linePoint1y, linePoint2x, linePoint2y))
		{
			if (cy1 > cy2)
			{
				count++;
			}
		}
		else if (IsPointOnLine(cx2, cy2, linePoint1x, linePoint1y, linePoint2x, linePoint2y))
		{
			if (cy2 > cy1)
			{
				count++;
			}
		}
		else if (IsIntersect(cx1, cy1, cx2, cy2, linePoint1x, linePoint1y, linePoint2x, linePoint2y))
		{
			count++;
		}
	}
	
	if(count % 2 == 1)
    {
        isInside = true;
    }
	return isInside;
}
/** #######################Judge whether Point in Polygon 2D#####################################**/


void violation_det(char *vdo_path)
{
    printf("#################Start violation detection######################\n");
    VideoCapture capture;
    Mat v_frame;
    capture >> v_frame;
    capture.open(vdo_path);
    if(!capture.isOpened())
    {
        printf("can not open ...\n");
        return;
    }
    int width = capture.get(CAP_PROP_FRAME_WIDTH);
    int height = capture.get(CAP_PROP_FRAME_HEIGHT);

    int obj_id,bx,by,bw,bh,redlight,frame_id,cur_frame,i,j,p;
    char classname[100];
    bool pedm_vio[MAX_FRAME];
    bool crline_vio[999];
    memset(zone, 0, sizeof(zone));
    memset(pedm_vio, 0, sizeof(pedm_vio));
    memset(crline_vio, 0, sizeof(crline_vio));

    rapidjson::Value violation(rapidjson::kArrayType);
    rapidjson::Document vOut(rapidjson::kObjectType);

    // detect speeding car
    for(i=0;i<spding_cnt;i++)
    {
        rapidjson::Value obj(rapidjson::kObjectType);
        obj.AddMember("obj_id", speedingBuffer[i].obj_id, vOut.GetAllocator());
        obj.AddMember("frame", speedingBuffer[i].frame_id, vOut.GetAllocator());
        obj.AddMember("vio_type", 4, vOut.GetAllocator());
        violation.PushBack(obj, vOut.GetAllocator());
        printf("车辆 %d 在第%d帧超速行驶！！！！！！\n", vzeroBuffer[i].obj_id, vzeroBuffer[i].frame_id);
    }

    // detect parked car in no-park ereas
    for(i=0;i<vzero_cnt;i++)
    {
        for(p=0; p<npark_num; p++)
        {
            if(Point_In_Polygon_2D(vzeroBuffer[i].bbox.x+vzeroBuffer[i].bbox.width/2,vzeroBuffer[i].bbox.y+vzeroBuffer[i].bbox.height/2,ctrs[p]))
            {
                rapidjson::Value obj(rapidjson::kObjectType);
                obj.AddMember("obj_id",vzeroBuffer[i].obj_id,vOut.GetAllocator());
                obj.AddMember("frame",vzeroBuffer[i].frame_id,vOut.GetAllocator());
                obj.AddMember("vio_type",5,vOut.GetAllocator());
                violation.PushBack(obj, vOut.GetAllocator());
                printf("车辆 %d 在第%.1fs违规停车！！！！！！\n", vzeroBuffer[i].obj_id, vzeroBuffer[i].frame_id*1.0/29.9);
                break;
            }
        }
        contours.clear();
    }

    // detect parked car in sidewalk ereas
    for(p=npark_num; p<npark_num+SidewalkNum; p++)
    {
        for(i=0;i<vzero_cnt;i++)
        {
            if(Point_In_Polygon_2D(vzeroBuffer[i].bbox.x+vzeroBuffer[i].bbox.width/2,vzeroBuffer[i].bbox.y+vzeroBuffer[i].bbox.height/2,ctrs[p]))
            {
                rapidjson::Value obj(rapidjson::kObjectType);
                obj.AddMember("obj_id",vzeroBuffer[i].obj_id,vOut.GetAllocator());
                obj.AddMember("frame",vzeroBuffer[i].frame_id,vOut.GetAllocator());
                obj.AddMember("vio_type",6,vOut.GetAllocator());
                violation.PushBack(obj, vOut.GetAllocator());
                printf("车辆 %d 在第%.1fs在人行道停车！！！！！！\n", vzeroBuffer[i].obj_id, vzeroBuffer[i].frame_id*1.0/29.9);
            }
        }
        contours.clear();
    }

    printf("####################pedstrian related ########################\n");
    // filter out persons on motor
    for(cur_frame=0; cur_frame<otherNodesList.size(); cur_frame++)
    {
        for(p=0; p<otherNodesList[cur_frame].size(); p++)
        {
            // cur_frame  otherNodesList[cur_frame][p]
            if(otherNodesList[cur_frame][p].getClassName()!="person") continue;
            for(int shift=max(cur_frame-64,0); shift<MIN(cur_frame+64,otherNodesList.size()); shift++)
            {
                for(j=0; j<otherNodesList[shift].size(); j++)
                {
                    if(otherNodesList[shift][j].getClassName()!="motor") continue;
                    if(abs(otherNodesList[cur_frame][p].getBbox().x-otherNodesList[shift][j].getBbox().x)<width*0.1
                        && abs(otherNodesList[cur_frame][p].getBbox().y-otherNodesList[shift][j].getBbox().y)<height*0.1 )
                    {
                        otherNodesList[cur_frame][p].setClassName("motor");
                        break;
                    }
                }
            }
        }
    }

    for(int i=1; i<otherNodesList.size(); i++)
    {
        for(j=0; j<otherNodesList[i].size(); j++)
        {
            if(otherNodesList[i][j].getClassName()!="person")
            {
                continue;
            }
            frame_id = i;
            bx = otherNodesList[i][j].getBbox().x+otherNodesList[i][j].getBbox().width/2;
            by = otherNodesList[i][j].getBbox().y+otherNodesList[i][j].getBbox().height/2;
            if(SidewalkNum==0)
            {
                pedm_vio[frame_id]=1;
                if(!pedm_vio[frame_id-1]) continue;
                rapidjson::Value obj(rapidjson::kObjectType);
                obj.AddMember("obj_id",-1,vOut.GetAllocator());
                obj.AddMember("frame",frame_id,vOut.GetAllocator());
                obj.AddMember("vio_type",8,vOut.GetAllocator());
                violation.PushBack(obj, vOut.GetAllocator());
                printf("当前帧数:%d   行人横穿马路!!!\n", frame_id);
            }
  
            for(int p=npark_num; p<npark_num+SidewalkNum; p++)
            {
                if(Point_In_Polygon_2D(bx,by,ctrs[p]))
                {
                    pedm_vio[frame_id]=1;
                    if(!pedm_vio[frame_id-1]) continue;
                    rapidjson::Value obj(rapidjson::kObjectType);
                    obj.AddMember("obj_id",-1,vOut.GetAllocator());
                    obj.AddMember("frame",frame_id,vOut.GetAllocator());
                    obj.AddMember("vio_type",8,vOut.GetAllocator());
                    violation.PushBack(obj, vOut.GetAllocator());
                    printf("当前帧数:%d   行人横穿马路!!!\n", frame_id);
                    break;
                }else if(car_light!=0 && is_red[frame_id][0]!=1){
                    pedm_vio[frame_id]=1;
                    if(!pedm_vio[frame_id-1]) continue;
                    rapidjson::Value obj(rapidjson::kObjectType);
                    obj.AddMember("obj_id",-1,vOut.GetAllocator());
                    obj.AddMember("frame",frame_id,vOut.GetAllocator());
                    obj.AddMember("vio_type",3,vOut.GetAllocator());
                    violation.PushBack(obj, vOut.GetAllocator());
                    printf("当前帧数:%d   行人闯红灯!!!\n", frame_id);
                    break;
                }
                contours.clear();
            }
        }

    }

    printf("####################  pedstrian end ########################\n");

    printf("###################### car #################################\n");
    // car related
    for(int i=0;i<trkNodesList.size();i++)
    {
        for(int j=0;j<trkNodesList[i].size();j++)
        {
            obj_id = trkNodesList[i][j].getObjId();
            frame_id = trkNodesList[i][j].getFrmId();
            bx = trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width/2;
            by = trkNodesList[i][j].getBbox().y+trkNodesList[i][j].getBbox().height/2;

            if(zone[obj_id]==0){
                for(p=0; p<lane_num; p++)
                {
                    if(Point_In_Polygon_2D(bx,by,lane[p]))
                    {
                        zone[obj_id]=p+1;
                        break;
                    }                    
                }
            }else{
                for(p=0; p<lane_num; p++)
                {
                    if(p+1==zone[obj_id]){
                        continue;
                    }
                    if(Point_In_Polygon_2D(bx,by,lane[p]))
                    {
                        crline_vio[frame_id]=1;
                        if(!crline_vio[max(frame_id-1,0)] | !crline_vio[max(frame_id-2,0)]) continue;
                        rapidjson::Value obj(rapidjson::kObjectType);
                        obj.AddMember("obj_id",obj_id,vOut.GetAllocator());
                        obj.AddMember("frame",frame_id,vOut.GetAllocator());
                        obj.AddMember("vio_type",1,vOut.GetAllocator());
                        violation.PushBack(obj, vOut.GetAllocator());
                        printf("当前时间:%.1fs   %d->%d 车辆 %d 出现违规越线行为!!!\n", (trkNodesList[i][j].getFrmId()*1.0)/29.9, zone[obj_id], p+1,obj_id);                        
                        break;
                    }
                }
                
                if(!stopline)
                {
                    continue;
                }
                if(by<(stop_line.k*bx+stop_line.b)-10)
                {
                    if(!(zone[i]==1 && is_red[frame_id][0]==1))
                    {
                        continue;
                    }
                    if(!(zone[i]!=1 && is_red[frame_id][1]==1))
                    {
                        continue;
                    }
                    rapidjson::Value obj(rapidjson::kObjectType);
                    obj.AddMember("obj_id",obj_id,vOut.GetAllocator());
                    obj.AddMember("frame",frame_id,vOut.GetAllocator());
                    obj.AddMember("vio_type",2,vOut.GetAllocator());
                    violation.PushBack(obj, vOut.GetAllocator());
                    printf("当前帧数:%d   车辆 %d 出现闯红灯行为!!!\n", frame_id, obj_id);                
                }

            }
        }
    }


    vOut.AddMember("violation", violation, vOut.GetAllocator());

    FILE* fp = fopen("violation.json", "wb");
    rapidjson::StringBuffer oBuffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> oWriter(oBuffer);

    vOut.Accept(oWriter);
    fputs(oBuffer.GetString(), fp);
    fclose(fp);
    printf("#################End violation detection######################\n");
}

bool check_speed(const vector<TrackNode>& objnodes, size_t from, size_t to, float spdthd){
    for(auto obj = objnodes.begin() + from; obj <= objnodes.begin() + to; obj++){
        if(obj->getSpd() <= spdthd) return false;
    }
    return true;
}


LIB_API void speed_estimation(char *json_path) {
    FILE *fprogress = fopen("./progress.app", "w");
    fprintf(fprogress, "2/0/1");
    fclose(fprogress);

    Rounder rounder(1);

    Settings s(json_path);

    // read in notation info
    FILE *f_nota = fopen(s.not_path,"r");
    if(f_nota == nullptr){
        fputs("Notation File Not Found!\n", stderr);
        return;
    }
    printf("##################Start reading notation file################\n");
    fscanf(f_nota, "%d %d %d %d %d %d", &lane_num, &stopline, &car_light, &SidewalkNum, &npark_num, &npark_v);

    int a,b,c,d,p,i=0;
    while( i<lane_num )  // read-in traffic line
    {
        for(int j=0;j<4;j++)
        {
            fscanf(f_nota,"%d %d ",&a,&b);
 	        lane[i].push_back(cv::Point(a,b));           
        }
        i++;
    }

    for(i=0; i<car_light; i++)                             // read-in traffic light
    {
        fscanf(f_nota, "%d %d %d %d", &a, &b, &c, &d);
        redlight[i].push_back(cv::Point(a,b));
        redlight[i].push_back(cv::Point(c,d));
    }

    if(stopline)
    {
        fscanf(f_nota, "%d %d %d %d", &a, &b, &c, &d );   // read-in stop line
    }
    stop_line.k = (d-b)*1.00 / (c-a+0.01);
    stop_line.b = b - stop_line.k * a;
    
    contours.clear();
    if(npark_num!=0)                                       // read-in noparking zones
    {
        for(i=0;i<npark_num;i++)
        {
            for(p=0;p<npark_v;p++)
            {
                fscanf(f_nota, "%d %d ", &a, &b);
                ctrs[i].push_back(cv::Point(a,b));
            }
            contours.push_back(ctrs[i]);
        }
    }

    if(SidewalkNum!=0)                                    // read-in sidewalk zones
    {
        for(i=npark_num; i<SidewalkNum+npark_num; i++)
        {

            for(p=0;p<4;p++)
            {
                fscanf(f_nota, "%d %d ", &a, &b);
                ctrs[i].push_back(cv::Point(a,b));
            }
            contours.push_back(ctrs[i]);
        }
    }

    fclose(f_nota);
    printf("##################notation file closed######################## \n");

    cv::VideoCapture cap(s.in_vdo_path);
    if (!cap.isOpened()) {
        throw FileNotFound(s.in_vdo_path);
    }
    s.frm_rate = cap.get(CV_CAP_PROP_FPS);
    s.frm_size.width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    s.frm_size.height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    auto totalFrmCnt = cap.get(CV_CAP_PROP_FRAME_COUNT);

    cv::VideoWriter vdoWriter;
    if (s.output_vdo_flag) {
        vdoWriter = cv::VideoWriter(s.out_vdo_path, CV_FOURCC('H', '2', '6', '4'), s.frm_rate, s.frm_size);
    }
    otherNodesList.reserve(int(totalFrmCnt) + 10);
    FILE *fpCam = fopen(s.cam_cal_path, "r");
    if (fpCam == NULL) {
        fputs("Camera Projection Matrix Not Loaded\n", stderr);
        return;
    }
    float proMat[12];
    char proBuffer[256];
    while (fgets(proBuffer, 256, fpCam) != NULL) {
        sscanf(proBuffer, "%f %f %f %f %f %f %f %f %f %f %f %f",
               &proMat[0], &proMat[1], &proMat[2], &proMat[3],
               &proMat[4], &proMat[5], &proMat[6], &proMat[7],
               &proMat[8], &proMat[9], &proMat[10], &proMat[11]);
    }
    fclose(fpCam);
    //! Tracking Format
    //! <frame_id>,<obj_id>,<xmin>,<ymin>,<width>,<height>,<confidence>,-1,-1,<class> (-1 if not tracking needed)
    cv::Rect bbox;
    cv::Point2f point2d;
    cv::Point3f point3d;
    char typeclass[32];
    memset(flow_count,0,sizeof(flow_count));
    memset(obj_set,0,sizeof(obj_set));
    TrackNode trkNode;
    int frame_id, obj_id, iii;
    float detConf = 0.0, depth = 0.0;
    FILE *fpTrk = fopen(s.in_trk_path, "r");
    if (fpTrk == NULL) {
        fputs("Track 2D File does not exist! \n", stderr);
        return;
    }
    char trkBuffer[512];
    while (fgets(trkBuffer, 512, fpTrk) != NULL) {
        sscanf(trkBuffer, "%d,%d,%d,%d,%d,%d,%f,1,%f,%f,%s",
               &frame_id, &obj_id, &bbox.x, &bbox.y,
               &bbox.width, &bbox.height, &detConf, &point3d.x,
               &point3d.y, typeclass);
        if(obj_id == -1){
            while(frame_id + 1 > otherNodesList.size()){
                otherNodesList.emplace_back(vector<OtherNode>());
            }
            otherNodesList[frame_id].emplace_back(OtherNode(typeclass, detConf, bbox));
            continue;
        }
        if(obj_set[obj_id]==0)
        {
            obj_set[obj_id]=1;
            for(iii=frame_id;iii<MAX_FRAME;iii++)
            {
                flow_count[iii]++;
            }

        }
        point2d = cv::Point2d(bbox.x + (bbox.width / 2.0f), bbox.y + (bbox.height / 2.0f));
        point3d = TrackNode::pro2dto3d(point2d, proMat);
        depth = cv::norm(point3d);
        trkNode = TrackNode(frame_id, obj_id, bbox, detConf, point2d, point3d, typeclass, depth);
        while (trkNodesList.size() < (obj_id + 1)) {
            trkNodesList.emplace_back(vector<TrackNode>());
        }

        if (detConf > s.det_thld) {
            trkNodesList[obj_id].emplace_back(trkNode);
        }
    }
    fclose(fpTrk);
    // Following procedures compute speed via distance between every frame_gap
    int frame_cnt, start_idx, end_idx, spd_win_sz;
    float total_dist;
    for (auto &i : trkNodesList) { // For each object
        frame_cnt = i.size();
        // if less than frame_gap frames for a certain object
        if (frame_cnt < s.frame_gap) spd_win_sz = frame_cnt;
        else spd_win_sz = s.frame_gap;
        // if even spd_win_sz turn to non-even
        // to make sure start_idx is an integer
        if (spd_win_sz % 2 == 0) spd_win_sz -= 1;
        for (int j = 0; j < frame_cnt; j++) { // For each frame of an object
            total_dist = 0.0;
            start_idx = j - ((spd_win_sz - 1) / 2);
            if (start_idx < 0) start_idx = 0;
            end_idx = j + ((spd_win_sz - 1) / 2);
            if (end_idx >= frame_cnt) end_idx = frame_cnt - 1;

            for (int k = start_idx; k < end_idx; k++) {  // accumulate distance between start_idx and end_idx
                total_dist += cv::norm(i[k].getPoint3D() - i[k + 1].getPoint3D());
            }
            i[j].setSpd(total_dist * s.frm_rate * s.speed_scale * 3.6f / (end_idx - start_idx));
        }
    }
    // adjustments

    // 1. For all or 1/3 frames that are close to camera, we compute the mean value of speed separately.
    vector<double> spd_mean, spd_std, spd_mean_cls;
    double spd_const_avg = 0.0;
    int spd_const_avg_num = 0;
    for (int i = 0; i < trkNodesList.size(); i++) {  // For each object
        frame_cnt = trkNodesList[i].size();
        spd_mean.push_back(0.0);
        spd_mean_cls.push_back(0.0);

        if (frame_cnt > s.frame_gap) {
            if (frame_cnt > 1)
                std::sort(trkNodesList[i].begin(), trkNodesList[i].end(), cmpDepth);  // all frames sorted by depth
            for (int j = 0; j < frame_cnt; j++) {
                spd_mean[i] += trkNodesList[i][j].getSpd(); // accumulate each frame's speed
                if (((frame_cnt / 3) + (s.frame_gap / 2) > j) &&
                    (s.frame_gap / 2 <= j)) { // those 1/3 frames that are close to camera
                    spd_mean_cls[i] += trkNodesList[i][j].getSpd();
                }
            }

            spd_mean[i] /= frame_cnt; // get mean value
            spd_mean_cls[i] /= (frame_cnt / 3);
        }
    }


    for (int i = 0; i < trkNodesList.size(); i++) {
        spd_std.push_back(0.0);
        frame_cnt = trkNodesList[i].size();
        if (frame_cnt > s.frame_gap) {
            if (s.spd_low_bound < spd_mean_cls[i]) {  // if special mean speed faster than low_bound
                for (int j = 0; j < frame_cnt; j++) {  // compute each obj std
                    spd_std[i] +=
                            (trkNodesList[i][j].getSpd() - spd_mean[i]) * (trkNodesList[i][j].getSpd() - spd_mean[i]);
                }
                spd_std[i] /= frame_cnt;
                spd_std[i] = std::sqrt(spd_std[i]);
                if (s.spd_std_thld > spd_std[i]) { // if small variance
                    for (int j = 0; j < frame_cnt; j++) {
                        trkNodesList[i][j].setSpd(spd_mean_cls[i]);
                        spd_const_avg += spd_mean_cls[i];
                        spd_const_avg_num += 1;
                    }
                }
            }

            for (int j = 0; j < frame_cnt; j++) {  // if spd smaller than stop_threshold
                if (s.spd_stop_thld > trkNodesList[i][j].getSpd()) {
                    trkNodesList[i][j].setSpd(0);
                }
            }

            std::sort(trkNodesList[i].begin(), trkNodesList[i].end(), cmpFrm);
        }
    }

    // remove those small obj_id
    trkNodesList.erase(remove_if(trkNodesList.begin(), trkNodesList.end(), [s](const vector<TrackNode> &obj) {
        return obj.size() <= s.frame_thld;
    }), trkNodesList.end());



    // ready to show results
    char str_obj_id[32];
    char str_f_c[100];
    char str_spd[32];
    int trj_len = 0;
    cv::Mat oriImage, carROI;
    pr::PlateWrapper pr(s.buffer);
    // compute carPlate & carType
    Vehicle vehicles[trkNodesList.size()]; // all objects
//    for(int i = 0; i < trkNodesListDepth.size(); i++){
//        frame_cnt = trkNodesListDepth[i].size();
//        if(frame_cnt == 0) continue;  // an object never appears
//
//        for(int j = 0; j < trkNodesListDepth[i].size(); j += )
//    }

    // Freetype Related
    cv::Ptr<cv::freetype::FreeType2> ft2;
    cv::Size textSize;
    cv::String text;
    cv::Point ptr_f_c;
    cv::Scalar base_c;
    ptr_f_c.x = s.frm_size.width * 0.06;
    ptr_f_c.y = s.frm_size.height * 0.06;
    int nota_y;
    int fontHeight = int(s.frm_size.height * 0.0125);
    int thickness = 0;
    int baseline = 0;
    ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData("msyh.ttf", 0);

    VID vid(s.buffer);
    namedWindow("Speed_estimation", CV_GUI_EXPANDED);
    resizeWindow("Speed_estimation", 640, 480);

    FILE *f_veh = fopen("vehicle.txt", "w+");
    if (f_veh == NULL) {
        fputs("vehicle.txt File open failed! \n", stderr);
        return;
    }

    FILE *f_rl = fopen(s.traffic_path,"r");
    short last_status[8]= {0};
    if(f_rl == nullptr){
        fputs("[In violation_detection func] Redlight File Not Found! \n", stderr);
        return;
    }

    int fr = 0;
    while(fr<totalFrmCnt-1)
    {
        fscanf(f_rl, "%d ", &fr);
        for(int p=0; p<car_light; p++)
        {
            fscanf(f_rl, "%d ", &is_red[fr][p]);
            if(is_red[fr][p]!=0)
            {
                last_status[p] = is_red[fr][p];
            }else{
                is_red[fr][p] = last_status[p];
            }
        }
    }

    fr=0;
    while (cap.read(oriImage)) {  // for each frame
        fr++;

        // draw other nodes
        if(!otherNodesList[fr].empty()){
            for(auto & obj : otherNodesList[fr]){
                cv::rectangle(oriImage, obj.getBbox(), Scalar(85, 153, 0), 2);
                cv::putText(oriImage, obj.getClassName().c_str(), obj.getPoint(), cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(85, 153, 0), 3, 8, 0);
            }
        }

        // Rendering no-park zones && sidewalk zones && lanes && light status
        for(int p=0; p<SidewalkNum+npark_num; p++)
        {
            contours.push_back(ctrs[p]);
	        cv::polylines(oriImage, contours, true, cv::Scalar(0,0,255), 2, cv::LINE_AA);
            contours.clear();
        }

        for(int p=0;p<lane_num;p++)
        {
            contours.push_back(lane[p]);
            cv::polylines(oriImage, contours, true, cv::Scalar(220,220,220), 2, cv::LINE_AA);
            contours.clear();
        }

        for(int i=0; i<car_light; i++)
        {
            switch(is_red[fr][i]){
                case 1:
    	            cv::rectangle(oriImage, redlight[i][0], redlight[i][1], Scalar(0,0,255), 1, 1, 0);
                    break;
                case 2:
    	            cv::rectangle(oriImage, redlight[i][0], redlight[i][1], Scalar(0,255,0), 1, 1, 0);
                    break;
                case 3:
                    cv::rectangle(oriImage, redlight[i][0], redlight[i][1], Scalar(193,37,255), 1, 1, 0);
                    break;
            }
        }
        // Rendering end

        sprintf(str_f_c, "FlowCount %d ", flow_count[fr]);
        cv::putText(oriImage, str_f_c, ptr_f_c, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(205,0,102), 3, 8, 0);        // draw flow-count
        for (int i = 0; i < trkNodesList.size(); i++) { // draw each object
            frame_cnt = trkNodesList[i].size();
            if (frame_cnt == 0) continue;  // an object never appears
            int last=0,isRedRc;
            if (fr >= trkNodesList[i][0].getFrmId() &&
                fr <= trkNodesList[i][frame_cnt - 1].getFrmId()) {  // if in the range of object i
                for (int j = 0; j < frame_cnt; j++) {
                    if(fr != trkNodesList[i][j].getFrmId()) continue;
                    isRedRc=0;
                    // 超速判断
                    if(trkNodesList[i][j].getSpd()>=s.spd_std_thld
                        && check_speed(trkNodesList[i], j, j + int(1 * s.frm_rate), s.spd_std_thld))
                    {
                        isRedRc=1;
                        speedingBuffer[spding_cnt].obj_id   = trkNodesList[i][j].getObjId();
                        speedingBuffer[spding_cnt].frame_id = trkNodesList[i][j].getFrmId();
                        spding_cnt++;
                        // printf("speeding>>> %f %f\n",trkNodesList[i][j].getSpd(),s.spd_std_thld);
                    }

                    // 停车判断
                    if(trkNodesList[i][j].getSpd()==0
                        &&trkNodesList[i][last].getSpd()==0)
                    {
                        isRedRc=1;
                        vzeroBuffer[vzero_cnt].obj_id   = trkNodesList[i][j].getObjId();
                        vzeroBuffer[vzero_cnt].frame_id = trkNodesList[i][j].getFrmId();
                        vzeroBuffer[vzero_cnt].bbox = trkNodesList[i][j].getBbox();
                        vzero_cnt++;
                        // printf("Parking>>> %f  obj %d at frame %d\n", trkNodesList[i][j].getSpd(), trkNodesList[i][j].getObjId(), fr);
                    }

                    // output rendered video
                    if (s.output_vdo_flag || s.show_when_computing) {

                        cv::circle(oriImage, cv::Point2d(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width/2,trkNodesList[i][j].getBbox().y+trkNodesList[i][j].getBbox().height/2),6, cv::Scalar(0,0,255),thickness=2);
                        if(isRedRc){
                            base_c = Scalar(0,0,255);
                        }else{
                            base_c = Scalar(0,205,102);
                        }
                        cv::rectangle(oriImage, trkNodesList[i][j].getBbox(), base_c, 2);
                        sprintf(str_obj_id, "ID: %d", i + 1);
                        cv::putText(oriImage, str_obj_id, trkNodesList[i][j].getPoint2D(), cv::FONT_HERSHEY_SIMPLEX, 1,
                                    Scalar(209,206,0), 2,CV_AA);

                        // put_text of detail info for each car
                        nota_y = trkNodesList[i][j].getBbox().y+20;
                        sprintf(str_spd, "Speed %.1fkm/h", trkNodesList[i][j].getSpd());
                        text = str_spd;
                        textSize = ft2->getTextSize(text, fontHeight, thickness, &baseline);
                        cv::rectangle(oriImage, cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width, nota_y),cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width+100, nota_y-18), base_c,-1);
                        ft2->putText(oriImage, text,
                                     cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width, nota_y),
                                     fontHeight, Scalar(0,0,0), thickness, CV_AA, true);

                        if(strlen(vehicles[i].plateInfo.c_str())>2){
                            nota_y+=20;
                            sprintf(str_spd, "Plate %s", vehicles[i].plateInfo.c_str());
                            text = str_spd;
                            textSize = ft2->getTextSize(text, fontHeight, thickness, &baseline);
                            cv::rectangle(oriImage, cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width, nota_y),cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width+100, nota_y-18), base_c,-1);
                            ft2->putText(oriImage, text,
                                         cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width, nota_y),
                                         fontHeight, Scalar(0,0,0), thickness, CV_AA, true);
                        }
                        if(strlen(vehicles[i].carType.c_str())>2){
                            nota_y+=20;
                            sprintf(str_spd, "Type %s", trkNodesList[i][j].getClasstype());
                            text = str_spd;
                            textSize = ft2->getTextSize(text, fontHeight, thickness, &baseline);
                            cv::rectangle(oriImage, cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width, nota_y),cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width+100, nota_y-18), base_c,-1);
                            ft2->putText(oriImage, text,
                                         cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width, nota_y),
                                         fontHeight, Scalar(0,0,0), thickness, CV_AA, true);
                        }
                        nota_y+=20;
                        sprintf(str_spd, "Conf %.1f%%", trkNodesList[i][j].getDetConf()*100.0);
                        text = str_spd;
                        textSize = ft2->getTextSize(text, fontHeight, thickness, &baseline);
                        cv::rectangle(oriImage, cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width, nota_y),cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width+100, nota_y-18), base_c,-1);
                        ft2->putText(oriImage, text,
                                     cv::Point(trkNodesList[i][j].getBbox().x+trkNodesList[i][j].getBbox().width,
                                               nota_y),
                                     fontHeight, Scalar(0,0,0), thickness, CV_AA, true);


                        /*
                        trj_len = std::min(s.traj_length, j + 1);
                        for (int k = j; k > j - trj_len + 1; k--) {
                            cv::line(oriImage, trkNodesList[i][k].getPoint2D(), trkNodesList[i][k - 1].getPoint2D(),
                                     vBoxColor[i % vBoxColor.size()], 2);
                        }
                        */
                        if (fr % s.frame_gap == 0) {
                            // printf("DEBUG: Obj_id: %d, Max Delta Depth: %.2f\n", i, trkNodesListDepth[i][frame_cnt - 1].getDepth() - trkNodesListDepth[i][0].getDepth()); // for DEBUG
                            carROI = ImageCropPadding(oriImage, trkNodesList[i][j].getBbox());
                            pr.DetectMat(carROI);
                            if (!pr.getRes().empty()) {
                                auto topPlate = pr.getRes().begin();
                                if (vehicles[i].plateConf < topPlate->confidence) { // choose the one with highest result
                                    string wait_str = plateFilter(
                                            topPlate->getPlateName());  // only allow those plates with legal string
                                    if (!wait_str.empty()) vehicles[i].plateInfo = wait_str;
                                    vehicles[i].plateConf = topPlate->confidence;
                                    vehicles[i].plateBBox = topPlate->getPlateRect();
                                    vehicles[i].plateBBox += cv::Point(trkNodesList[i][j].getBbox().x,
                                                                       trkNodesList[i][j].getBbox().y);
                                    vehicles[i].carBBox = trkNodesList[i][j].getBbox();
                                }
                            }
                            if (vid.enabled) {
                                auto res = vid.recognize(carROI);
                                if (vehicles[i].typeConf < res.second) {
                                    vehicles[i].carType = res.first;
                                    vehicles[i].typeConf = res.second;
                                }
                            }

                        }

                        // break;  // once found
                    }
                }

            }

        }

        if (s.show_when_computing) {
            cv::imshow("Speed_estimation", oriImage);
            cv::waitKey(1);
        }
        if (s.output_vdo_flag) {
            vdoWriter.write(oriImage);
        }
    }

    for(int i=0; i<trkNodesList.size(); i++)
    {
        fprintf(f_veh, "%d %d %d %d %d %d %s\n", i, vehicles[i].carBBox.x, vehicles[i].carBBox.y, vehicles[i].carBBox.height, vehicles[i].carBBox.width, vehicles[i].plateInfo);
    }
    fclose(f_veh);
    fclose(f_rl);
    rapidjson::Document dOut(rapidjson::kObjectType);
    rapidjson::Value general(rapidjson::kArrayType);
    /**
     *  ################ example ####################
     *  "obj_id": 1,
     *  "plate": "川AXXXXX",
     *  "avg_speed": 34.22,
     *  "car_type": "东风日产",
     *  "car_color": "红",
     *  "st_frame": 123,
     *  "ed_frame": 456
     *  ############################################
     **/

    for (int i = 0; i < trkNodesList.size(); ++i) { // general generated
        if (trkNodesList[i].empty()) continue;
        rapidjson::Value speed(rapidjson::kArrayType);
        for (int j = 0; j < trkNodesList[i].size(); ++j) {
            rapidjson::Value speed_frame(rapidjson::kObjectType);
            speed_frame.AddMember("frm", trkNodesList[i][j].getFrmId(), dOut.GetAllocator());
            speed_frame.AddMember("spd", trkNodesList[i][j].getSpd(), dOut.GetAllocator());
            speed.PushBack(speed_frame, dOut.GetAllocator());
        }

        rapidjson::Value obj(rapidjson::kObjectType);
        obj.AddMember("obj_id", i + 1, dOut.GetAllocator());
        obj.AddMember("plate", vehicles[i].plateInfo, dOut.GetAllocator());
        double totalSum = std::accumulate(trkNodesList[i].begin(), trkNodesList[i].end(), 0.0, SpeedSumHelper());
        obj.AddMember("avg_speed", (int) ((totalSum / trkNodesList[i].size()) * 100) / 100.0, dOut.GetAllocator());
        obj.AddMember("speed", speed, dOut.GetAllocator());

        obj.AddMember("car_type", vehicles[i].carType, dOut.GetAllocator());
        obj.AddMember("car_color", vehicles[i].carType, dOut.GetAllocator());  //TODO: Color not finished
        obj.AddMember("st_frame", trkNodesList[i][0].getFrmId(), dOut.GetAllocator());
        obj.AddMember("ed_frame", trkNodesList[i][trkNodesList[i].size() - 1].getFrmId(), dOut.GetAllocator());
        general.PushBack(obj, dOut.GetAllocator());
    }

    dOut.AddMember("general", general, dOut.GetAllocator());

    FILE *fp = fopen("general.json", "wb");
    rapidjson::StringBuffer oBuffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> oWriter(oBuffer);
    oWriter.SetMaxDecimalPlaces(1);

    dOut.Accept(oWriter);
    fputs(oBuffer.GetString(), fp);
    fclose(fp);
    cv::destroyAllWindows();

    violation_det(s.in_vdo_path);
    vdoWriter.release();
    fprogress = fopen("./progress.app", "w");
    fprintf(fprogress, "2/1/1");
    fclose(fprogress);
}

const string &OtherNode::getClassName() const {
    return class_name;
}

void OtherNode::setClassName(const string &className) {
    class_name = className;
}

float OtherNode::getDetConf() const {
    return det_conf;
}

void OtherNode::setDetConf(float detConf) {
    det_conf = detConf;
}

const Rect &OtherNode::getBbox() const {
    return bbox;
}

void OtherNode::setBbox(const Rect &bbox) {
    OtherNode::bbox = bbox;
}

OtherNode::OtherNode(string className, float detConf, const Rect &bbox) : class_name(std::move(className)),
                                                                                 det_conf(detConf), bbox(bbox) {
    point.x = bbox.x + bbox.width / 2;
    point.y = bbox.y;
}

const Point2f &OtherNode::getPoint() const {
    return point;
}
