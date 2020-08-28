#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <unistd.h>

using namespace cv;
using namespace std;

Mat src, cloneimg;
bool mousedown;
vector<vector<Point> > contours;
vector<Point> pts;

Point Prev;

void myOnMouse(int event, int x, int y, int flags, void *userdata) {
    Mat img = *((Mat *) userdata);
    if (event == EVENT_LBUTTONDOWN) {
        circle(img, Point(x, y), 6, Scalar(255, 0, 0), 1, CV_AA);
        pts.push_back(Point(x, y));
        if (pts.size() > 1) {
            line(img, Prev, Point(x, y), Scalar(0, 255, 0));
        }
        Prev = Point(x, y);
    }
    imshow("Create Mask", img);
}

void drawROI(const cv::Mat &src, const char *out_mask_path) {
    if (src.empty()) {
        return;
    }

    namedWindow("Create Mask", WINDOW_NORMAL);
    cloneimg = src.clone();
    setMouseCallback("Create Mask", myOnMouse, &cloneimg);
    imshow("Create Mask", src);
    while (1) {
        int nKey = waitKey(0);
        if (nKey == 27) break;
        if (nKey == 'o') {  // confirm completing points selection
            if (pts.size() > 2) {
                Mat mask(cloneimg.size(), CV_8UC1);
                mask = 0;
                contours.push_back(pts);
                drawContours(mask, contours, 0, Scalar(255), -1);
                Mat masked(cloneimg.size(), CV_8UC3, Scalar(255, 255, 255));
                src.copyTo(masked, mask);
                src.copyTo(cloneimg);
                imshow("masked", masked);
                int _nKey = waitKey(0);
                if (_nKey == 'o') {   // confirm a suitable mask
                    imwrite(out_mask_path, mask);
                    break;
                } else {
                    destroyWindow("masked");
                    pts.clear();
                    contours.clear();
                    continue;
                }
            }
        }
    }
    cv::destroyAllWindows();
}


FILE *f_nota;
int   lane_num,stopline,car_light,npark_num,sw_num,npark_v;
Point pt[1][4];
int   npt[] = { 4 };
std::vector<cv::Point > contour;
void NotaOnMouse(int event, int x, int y, int flags, void *userdata) {
    Mat img = *((Mat *) userdata);
    if(pts.size()<lane_num*4){   // car lines
        printf("请在图中框选出监控区域的车道\n");
    }else if(car_light!=0 && pts.size()<lane_num*4+2*car_light) // red lights corresponding to car lines
    {
        printf("请在图中标注车道对应的红绿灯区块\n");
    }else if(stopline!=0 && pts.size()<lane_num*4+2*car_light+2*stopline){        // stop line
        printf("请在图中标注停车线\n");        
    }else if(npark_num!=0 && pts.size()<lane_num*4+2*car_light+2*stopline+npark_v*npark_num){      // no-park zones                  
        printf("请在图中标注禁止停车区块\n");
    }else if(sw_num!=0 && pts.size()<lane_num*4+2*car_light+2*stopline+npark_v*npark_num+4*sw_num){
        printf("请在图中标注人行道区块\n");
    }else{
        printf("标注完成，请按键盘o键退出\n");
    }

    if (event == EVENT_LBUTTONDOWN) {
        circle(img, Point(x, y), 3, Scalar(255, 0, 0), 1, 8);
        pts.push_back(Point(x, y));
        fprintf(f_nota, "%d %d ", x, y);

        if(pts.size()<=lane_num*4){   // car lines
 	        contour.push_back(cv::Point(x,y));           
            if(pts.size()%4==0)
            {
	            contours.push_back(contour);
	            cv::polylines(img, contours, true, cv::Scalar(0,0,255), 2, cv::LINE_AA);
                contour.clear();
                contours.clear();
                fprintf(f_nota, "\n");
            }
        }else if(car_light!=0 && pts.size()<=lane_num*4+2*car_light) // red lights corresponding to car lines
        {
            if((pts.size()-lane_num*4)%2==0){
                cv::rectangle (img, Prev, Point(x, y), Scalar(0, 255, 0), 1, 8, 0);
                fprintf(f_nota, "\n");
            }
        }else if(stopline!=0 && pts.size()<=lane_num*4+2*car_light+2*stopline){        // stop line
            if(pts.size()==lane_num*4+2*car_light+2)
            {
                line(img, Prev, Point(x, y), Scalar(0, 255, 0), 3);
                fprintf(f_nota, "\n");
            }
        }else if(npark_num!=0 && pts.size()<=lane_num*4+2*car_light+2*stopline+npark_v*npark_num){      // no-park zones                  
	        contour.push_back(cv::Point(x,y));            
            if((pts.size()-(lane_num*4+2*car_light+2*stopline))%npark_v==0)        
            {
	            contours.push_back(contour);
	            cv::polylines(img, contours, true, cv::Scalar(0,0,255), 2, cv::LINE_AA);
                contour.clear();
                contours.clear();
                fprintf(f_nota, "\n");
            }
        }else if(sw_num!=0 && pts.size()<=lane_num*4+2*car_light+2*stopline+npark_v*npark_num+4*sw_num){      // side-walk zones                  
	        contour.push_back(cv::Point(x,y));
            if((pts.size()-(lane_num*4+2*car_light+2*stopline+npark_v*npark_num))%4==0)        
            {
	            contours.push_back(contour);
	            cv::polylines(img, contours, true, cv::Scalar(0,0,255), 2, cv::LINE_AA);
                contour.clear();
                contours.clear();
                fprintf(f_nota, "\n");
            }
        }

        Prev = Point(x, y);
    }
    imshow("Draw Notation", img);
}

void drawNotation(const cv::Mat &src, const char *out_not_path) {
    pts.clear();
    if (src.empty()) {
        return;
    }

    contours.clear();
    cloneimg = src.clone();
    namedWindow("Draw Notation", WINDOW_NORMAL);

    imshow("Draw Notation", src);

    f_nota = fopen(out_not_path, "w+");
    printf("请输入需要标注的车道块数量以及车辆指示红绿灯数量，格式如'2 1'\n");
    scanf("%d %d", &lane_num, &car_light);
    printf("是否有停车线需要标注？请输入0或1\n");
    scanf("%d", &stopline);
    printf("请输入需要标注的人行道区块数量\n");
    scanf("%d", &sw_num);
    printf("请输入禁止停车区块的数量\n");
    scanf("%d", &npark_num);
    if(npark_num!=0)
    {
        printf("使用几边形来适配框选禁停区呢？不得小于4\n");
        scanf("%d", &npark_v);
    }
    fprintf(f_nota, "%d %d %d %d %d %d\n", lane_num, stopline, car_light, sw_num, npark_num, npark_v );

    setMouseCallback("Draw Notation", NotaOnMouse, &cloneimg);
    std::string not_str;
    while (1) {
        int nKey = waitKey(0);
        if (nKey == 'o') { 
            break;
        }
    }
    fclose(f_nota);
    cv::destroyAllWindows();
    exit(-1);
}