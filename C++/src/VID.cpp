#include <rapidjson/document.h>
#include "VID.h"

VID::VID(const string &model_path) {
    const string car_list[] = {"DS_4S", "DS_5", "DS_5LS", "DS_5LS_2014款", "DS_5_2014款", "DS_6", "DS_6_2014款",
                               "GMC_SAVANA", "Jeep_大切诺基", "Jeep_大切诺基_2014款", "Jeep_指南者", "Jeep_指南者_2011款",
                               "Jeep_指南者_2012款", "Jeep_指南者_2013款", "Jeep_指南者_2015款", "Jeep_指挥官", "Jeep_牧马人", "Jeep_自由侠",
                               "Jeep_自由侠_2016款", "Jeep_自由光", "Jeep_自由光_2016款", "Jeep_自由客", "Jeep_自由客_2014款", "MG3",
                               "MG3_2011款", "MG3_2013款", "MG3_2014款", "MG5", "MG5_2012款", "MG6", "MG6_2010款",
                               "MG6_2012款", "MG6_2013款", "MG6_2014款", "MG7", "MG_3SW", "MG_名爵6新能源", "MG_名爵ZS",
                               "MG_名爵ZS_2017款", "MG_锐腾", "MG_锐腾_2015款", "MG_锐腾_2016款", "MG_锐行", "MINI", "MINI_2011款",
                               "MINI_2014款", "MINI_2016款", "MINI_Clubman", "MINI_Clubman_2011款", "MINI_Countryman",
                               "MINI_Countryman_2011款", "MINI_Coupe", "MINI_Paceman", "Smart_smart_forfour",
                               "Smart_smart_fortwo", "Smart_smart_fortwo_2012款", "Smart_smart_fortwo_2015款",
                               "WEY_VV5_2017款", "WEY_VV7_2017款", "一汽_佳宝V52", "一汽_佳宝V70_II代", "一汽_佳宝V75", "一汽_佳宝V80",
                               "一汽_威姿", "一汽_威志", "一汽_威志V2", "一汽_威志V2_Cross", "一汽_威志V5", "一汽_森雅M80", "一汽_森雅R7",
                               "一汽_森雅R7_2016款", "一汽_森雅S80", "一汽_骏派A70", "一汽_骏派D60", "一汽_骏派D60_2015款", "三菱_劲炫",
                               "三菱_劲炫ASX", "三菱_劲炫ASX_2013款", "三菱_君阁", "三菱_帕杰罗", "三菱_帕杰罗·劲畅", "三菱_欧蓝德", "三菱_欧蓝德EX劲界",
                               "三菱_欧蓝德_2016款", "三菱_欧蓝德经典", "三菱_祺智PHEV", "三菱_翼神", "三菱_翼神_2011款", "三菱_翼神_2012款",
                               "三菱_翼神_2013款", "三菱_蓝瑟", "三菱_风迪思", "三菱劲炫ASX", "三菱戈蓝", "三菱风迪思", "上汽大通_大通D90_2017款",
                               "上汽大通_大通G10", "上汽大通_大通RV80", "上汽大通_大通V80", "上汽大通大通G10", "东南DX3", "东南DX3_2016款", "东南DX7",
                               "东南DX7_2015款", "东南_V3菱悦", "东南_V3菱悦_2012款", "东南_V3菱悦_2014款", "东南_V3菱悦_2015款", "东南_V5菱致",
                               "东南_V5菱致_2015款", "东南_V6菱仕", "东南_希旺", "东风_帅客", "东风_景逸", "东风_景逸_2012款", "东风_风光330",
                               "东风_风光360", "东风_风度MX6", "东风_风行CM7", "东风_风行S500_2016款", "东风_风行SX6", "东风小康C37", "东风小康K07S",
                               "东风小康K07_II", "东风小康K17", "东风小康V27", "东风小康V29", "东风小康_东风风光330", "东风小康_东风风光330_2014款",
                               "东风小康_东风风光360", "东风小康_东风风光370", "东风小康_东风风光580", "东风小康_东风风光580_2016款", "东风小康_东风风光S560",
                               "东风小康_风光", "东风小康_风光580", "东风风度MX5", "东风风神A30", "东风风神A60", "东风风神AX3_2016款", "东风风神AX5",
                               "东风风神AX7", "东风风神AX7_2015款", "东风风神AX7_2016款", "东风风神H30", "东风风神L60", "东风风神S30",
                               "东风风行_景逸S50", "东风风行_景逸SUV", "东风风行_景逸X3", "东风风行_景逸X3_2014款", "东风风行_景逸X5",
                               "东风风行_景逸X5_2013款", "东风风行_景逸X5_2015款", "东风风行_景逸X5_2016款", "东风风行_景逸X6", "东风风行_景逸XV",
                               "东风风行_景逸XV_2015款", "东风风行_菱智", "东风风行_菱智_2013款", "东风风行_菱智_2014款", "东风风行_菱智_2015款",
                               "东风风行_菱智_2016款", "东风风行_菱智_2018款", "中华H230", "中华H320", "中华H330_2013款", "中华H530",
                               "中华H530_2011款", "中华V3_2015款", "中华V3_2016款", "中华V5", "中华V5_2012款", "中华_尊驰", "中华_骏捷",
                               "中华_骏捷FRV", "中华_骏捷FSV_2011款", "丰田", "丰田4Runner", "丰田86", "丰田C-HR", "丰田Fortuner",
                               "丰田Sienna", "丰田YARiS_L_致炫", "丰田_FJ酷路泽", "丰田_RAV4", "丰田_RAV4_2009款", "丰田_RAV4_2011款",
                               "丰田_RAV4_2012款", "丰田_RAV4_2013款", "丰田_RAV4_2015款", "丰田_RAV4_2016款",
                               "丰田_YARiS_L_致享_2017款", "丰田_YARiS_L_致炫", "丰田_YARiS_L_致炫_2014款", "丰田_YARiS_L_致炫_2015款",
                               "丰田_YARiS_L_致炫_2016款", "丰田_兰德酷路泽", "丰田_凯美瑞", "丰田_凯美瑞_2009款", "丰田_凯美瑞_2010款",
                               "丰田_凯美瑞_2011款", "丰田_凯美瑞_2012款", "丰田_凯美瑞_2013款", "丰田_凯美瑞_2015款", "丰田_凯美瑞_2016款", "丰田_卡罗拉",
                               "丰田_卡罗拉_2007款", "丰田_卡罗拉_2008款", "丰田_卡罗拉_2011款", "丰田_卡罗拉_2012款", "丰田_卡罗拉_2013款",
                               "丰田_卡罗拉_2014款", "丰田_卡罗拉_2016款", "丰田_卡罗拉_2017款", "丰田_坦途", "丰田_埃尔法", "丰田_威驰",
                               "丰田_威驰FS_2017款", "丰田_威驰_2014款", "丰田_威驰_2016款", "丰田_威驰_2017款", "丰田_普拉多", "丰田_普拉多_2010款",
                               "丰田_普拉多_2014款", "丰田_普拉多_2016款", "丰田_普锐斯", "丰田_汉兰达", "丰田_汉兰达_2009款", "丰田_汉兰达_2011款",
                               "丰田_汉兰达_2012款", "丰田_汉兰达_2015款", "丰田_皇冠", "丰田_皇冠_2015款", "丰田_红杉", "丰田_花冠", "丰田_花冠_2009款",
                               "丰田_花冠_2011款", "丰田_花冠_2013款", "丰田_逸致", "丰田_逸致_2011款", "丰田_逸致_2014款", "丰田_锐志",
                               "丰田_锐志_2010款", "丰田_锐志_2012款", "丰田_锐志_2013款", "丰田_雅力士", "丰田_雅力士_2011款", "丰田_雷凌",
                               "丰田_雷凌_2014款", "丰田_雷凌_2016款", "丰田_雷凌_2017款", "九龙_艾菲", "五十铃D-MAX", "五十铃_瑞迈", "五十铃_竞技者",
                               "五十铃皮卡", "五菱之光", "五菱之光_2010款", "五菱宏光", "五菱宏光_2010款", "五菱宏光_2013款", "五菱宏光_2014款",
                               "五菱宏光_2015款", "五菱征程", "五菱荣光", "五菱荣光V", "五菱荣光V_2016款", "五菱荣光_2011款", "众泰SR7_2016款",
                               "众泰SR9", "众泰T200", "众泰T300", "众泰T600", "众泰T600_2014款", "众泰T600_2015款", "众泰T600_2016款",
                               "众泰T600_Coupe_2017款", "众泰T700_2017款", "众泰Z300", "众泰Z500", "众泰Z700", "众泰_大迈X5",
                               "众泰_大迈X5_2015款", "众泰_大迈X7", "众泰大迈X7", "传祺GA3", "传祺GA3S新能源", "传祺GA3S视界", "传祺GA3S视界_2014款",
                               "传祺GA5", "传祺GA6", "传祺GA6_2016款", "传祺GA8", "传祺GM8", "传祺GS3", "传祺GS4", "传祺GS4_2015款",
                               "传祺GS4_2016款", "传祺GS4_2017款", "传祺GS5_2012款", "传祺GS5_2013款", "传祺GS5_2014款",
                               "传祺GS5_Super_2015款", "传祺GS7", "传祺GS8_2017款", "依维柯Power_Daily", "依维柯Turbo_Daily",
                               "依维柯_都灵", "依维柯宝迪", "保时捷718", "保时捷911", "保时捷Boxster", "保时捷Cayenne", "保时捷Cayman",
                               "保时捷Macan", "保时捷Panamera", "克莱斯勒300C", "克莱斯勒300S", "克莱斯勒_大捷龙", "兰博基尼Huracan", "凯翼C3",
                               "凯翼C3R", "凯翼X3", "凯迪拉克ATS-L", "凯迪拉克ATS-L_2014款", "凯迪拉克ATS-L_2016款", "凯迪拉克ATS_2014款",
                               "凯迪拉克CT6", "凯迪拉克CTS", "凯迪拉克SRX", "凯迪拉克SRX_2011款", "凯迪拉克SRX_2012款", "凯迪拉克SRX_2013款",
                               "凯迪拉克SRX_2015款", "凯迪拉克XT5", "凯迪拉克XT5_2016款", "凯迪拉克XT5_2018款", "凯迪拉克XTS", "凯迪拉克XTS_2013款",
                               "凯迪拉克XTS_2014款", "凯迪拉克XTS_2016款", "凯迪拉克_凯雷德", "凯迪拉克_赛威SLS", "别克GL6", "别克GL8",
                               "别克GL8_2011款", "别克GL8_2012款", "别克GL8_2013款", "别克GL8_2014款", "别克GL8_2015款", "别克GL8_2017款",
                               "别克_凯越", "别克_凯越_2008款", "别克_凯越_2011款", "别克_凯越_2013款", "别克_凯越_2015款", "别克_凯越旅行车", "别克_君威",
                               "别克_君威_2009款", "别克_君威_2010款", "别克_君威_2011款", "别克_君威_2012款", "别克_君威_2013款", "别克_君威_2014款",
                               "别克_君威_2015款", "别克_君威_2017款", "别克_君越", "别克_君越_2008款", "别克_君越_2010款", "别克_君越_2011款",
                               "别克_君越_2012款", "别克_君越_2013款", "别克_君越_2014款", "别克_君越_2016款", "别克_威朗_2015款", "别克_威朗_2016款",
                               "别克_威朗_2017款", "别克_威朗_2018款", "别克_昂科威", "别克_昂科威_2014款", "别克_昂科威_2016款", "别克_昂科威_2017款",
                               "别克_昂科拉", "别克_昂科拉_2013款", "别克_昂科拉_2014款", "别克_昂科拉_2015款", "别克_昂科拉_2017款", "别克_昂科雷",
                               "别克_林荫大道", "别克_英朗", "别克_英朗_2010款", "别克_英朗_2011款", "别克_英朗_2012款", "别克_英朗_2013款",
                               "别克_英朗_2014款", "别克_英朗_2015款", "别克_英朗_2016款", "别克_英朗_2017款", "别克_阅朗", "别克威朗", "力帆320",
                               "力帆330", "力帆520", "力帆620", "力帆720", "力帆X50", "力帆X60", "力帆_丰顺", "力帆_迈威", "北京BJ20_2016款",
                               "北京BJ40", "北京BJ40_2016款", "北京BJ80", "北京汽车E系列_2012款", "北京汽车E系列_2013款", "北汽_BJ212",
                               "北汽_战旗", "北汽_绅宝D20", "北汽_绅宝D50", "北汽_绅宝D50_2014款", "北汽_绅宝D70", "北汽_绅宝X25",
                               "北汽_绅宝X25_2015款", "北汽_绅宝X35", "北汽_绅宝X55_2016款", "北汽_绅宝X65_2015款", "北汽威旺306", "北汽威旺307",
                               "北汽威旺M20", "北汽威旺M20_2014款", "北汽威旺M30", "北汽威旺M35", "北汽威旺S50", "北汽幻速H2", "北汽幻速H2V",
                               "北汽幻速H3", "北汽幻速H3F", "北汽幻速S2", "北汽幻速S3", "北汽幻速S3L", "北汽幻速S3_2014款", "北汽幻速S5", "北汽幻速S6",
                               "北汽幻速S6_2016款", "北汽幻速S7", "北汽新能源_EC系列", "北汽新能源_EU系列", "北汽新能源_EV系列", "北汽新能源_EV系列_2016款",
                               "华普_海域", "华泰_圣达菲", "华泰_圣达菲5", "华泰_圣达菲经典", "华泰_宝利格", "华泰_路盛E70", "华泰圣达菲", "华颂7", "双环SCEO",
                               "双龙_享御", "双龙_柯兰多", "双龙_爱腾", "吉利EC8", "吉利GC7", "吉利GX2", "吉利GX7_2012款", "吉利GX7_2013款",
                               "吉利GX7_2014款", "吉利GX7_2015款", "吉利SC3", "吉利SX7", "吉利_博瑞_2015款", "吉利_博瑞_2016款",
                               "吉利_博瑞_2017款", "吉利_博越_2016款", "吉利_博越_2018款", "吉利_帝豪EV", "吉利_帝豪GL_2017款", "吉利_帝豪GL_2018款",
                               "吉利_帝豪GS_2016款", "吉利_帝豪_2014款", "吉利_帝豪_2015款", "吉利_帝豪_2016款", "吉利_帝豪_2017款",
                               "吉利_帝豪_2018款", "吉利_海景", "吉利_熊猫", "吉利_经典帝豪", "吉利_经典帝豪_2012款", "吉利_自由舰", "吉利_英伦C5",
                               "吉利_英伦TX4", "吉利_豪情SUV", "吉利_远景", "吉利_远景S1", "吉利_远景SUV", "吉利_远景SUV_2016款", "吉利_远景X3",
                               "吉利_远景_2015款", "吉利_远景_2016款", "吉利_远景_2017款", "吉利_金刚", "吉利_金刚_2016款", "吉利_金刚财富", "吉利_金鹰",
                               "吉利博瑞", "吉利博越", "吉利金刚2代", "启辰D50", "启辰D50_2013款", "启辰D60", "启辰R50", "启辰R50X",
                               "启辰R50_2013款", "启辰R50_2015款", "启辰T70", "启辰T70X", "启辰T70_2015款", "启辰T90", "启辰_晨风", "哈弗H1",
                               "哈弗H1_2015款", "哈弗H1_2016款", "哈弗H2", "哈弗H2_2014款", "哈弗H2_2016款", "哈弗H2_2017款", "哈弗H2s",
                               "哈弗H3", "哈弗H4", "哈弗H5", "哈弗H5_2012款", "哈弗H5_2013款", "哈弗H6_2011款", "哈弗H6_2012款",
                               "哈弗H6_2013款", "哈弗H6_2014款", "哈弗H6_2015款", "哈弗H6_2016款", "哈弗H6_2017款", "哈弗H6_2018款",
                               "哈弗H6_Coupe", "哈弗H6_Coupe_2015款", "哈弗H6_Coupe_2016款", "哈弗H7", "哈弗H7_2017款", "哈弗H8",
                               "哈弗H8_2015款", "哈弗H9", "哈弗H9_2015款", "哈弗M6", "哈飞_民意", "哈飞赛豹III", "哈飞骏意", "夏利", "夏利N5",
                               "夏利N7", "大众CC", "大众CC_2010款", "大众CC_2011款", "大众CC_2012款", "大众CC_2013款", "大众CC_2015款",
                               "大众CC_2016款", "大众Eos", "大众POLO", "大众POLO_2009款", "大众POLO_2011款", "大众POLO_2012款",
                               "大众POLO_2013款", "大众POLO_2014款", "大众POLO_2016款", "大众_2016款", "大众_Passat领驭",
                               "大众_Passat领驭_2009款", "大众_凌渡_2015款", "大众_凌渡_2017款", "大众_凌渡_2018款", "大众_凯路威", "大众_夏朗",
                               "大众_夏朗_2013款", "大众_宝来", "大众_宝来_2008款", "大众_宝来_2011款", "大众_宝来_2012款", "大众_宝来_2013款",
                               "大众_宝来_2014款", "大众_宝来_2015款", "大众_宝来_2016款", "大众_宝来_2017款", "大众_宝来_2018款", "大众_尚酷",
                               "大众_尚酷_2011款", "大众_帕萨特", "大众_帕萨特_2011款", "大众_帕萨特_2013款", "大众_帕萨特_2014款", "大众_帕萨特_2015款",
                               "大众_帕萨特_2016款", "大众_帕萨特_2017款", "大众_捷达", "大众_捷达_2010款", "大众_捷达_2013款", "大众_捷达_2015款",
                               "大众_捷达_2017款", "大众_朗境", "大众_朗行", "大众_朗行_2013款", "大众_朗行_2015款", "大众_朗逸", "大众_朗逸_2008款",
                               "大众_朗逸_2011款", "大众_朗逸_2013款", "大众_朗逸_2015款", "大众_朗逸_2017款", "大众_桑塔纳", "大众_桑塔纳_2013款",
                               "大众_桑塔纳_2015款", "大众_桑塔纳_2016款", "大众_桑塔纳志俊", "大众_桑塔纳志俊_2008款", "大众_桑塔纳经典", "大众_甲壳虫",
                               "大众_甲壳虫_2013款", "大众_甲壳虫_2014款", "大众_蔚揽(进口)", "大众_蔚领", "大众_蔚领_2017款", "大众_辉昂", "大众_辉腾",
                               "大众_迈特威", "大众_迈腾", "大众_迈腾_2009款", "大众_迈腾_2011款", "大众_迈腾_2012款", "大众_迈腾_2013款",
                               "大众_迈腾_2015款", "大众_迈腾_2016款", "大众_迈腾_2017款", "大众_迈腾_2018款", "大众_途安", "大众_途安_2011款",
                               "大众_途安_2013款", "大众_途安_2015款", "大众_途昂", "大众_途观", "大众_途观L", "大众_途观_2010款", "大众_途观_2012款",
                               "大众_途观_2013款", "大众_途观_2015款", "大众_途观_2016款", "大众_途锐", "大众_途锐_2011款", "大众_速腾",
                               "大众_速腾_2009款", "大众_速腾_2010款", "大众_速腾_2011款", "大众_速腾_2012款", "大众_速腾_2014款", "大众_速腾_2015款",
                               "大众_速腾_2017款", "大众_速腾_2018款", "大众_高尔夫", "大众_高尔夫_2010款", "大众_高尔夫_2011款", "大众_高尔夫_2012款",
                               "大众_高尔夫_2014款", "大众_高尔夫_2015款", "大众_高尔夫_2016款", "大众_高尔夫_2017款", "大众_高尔夫·嘉旅",
                               "大众_高尔夫·嘉旅_2016款", "大众途观_2013款", "奇瑞A1", "奇瑞A3", "奇瑞A5", "奇瑞E3_2013款", "奇瑞E3_2015款",
                               "奇瑞E5", "奇瑞E5_2012款", "奇瑞QQ", "奇瑞QQ3", "奇瑞X1", "奇瑞_旗云1", "奇瑞_旗云2", "奇瑞_旗云3", "奇瑞_瑞虎",
                               "奇瑞_瑞虎3_2014款", "奇瑞_瑞虎3_2015款", "奇瑞_瑞虎3_2016款", "奇瑞_瑞虎5", "奇瑞_瑞虎5_2014款", "奇瑞_瑞虎5_2016款",
                               "奇瑞_瑞虎5x", "奇瑞_瑞虎7", "奇瑞_瑞虎_2012款", "奇瑞_艾瑞泽3", "奇瑞_艾瑞泽5", "奇瑞_艾瑞泽5_2016款", "奇瑞_艾瑞泽7",
                               "奇瑞_艾瑞泽7e", "奇瑞_风云2", "奇瑞_风云2_2010款", "奇瑞_风云2_2013款", "奇瑞eQ1", "奇瑞瑞虎", "奇瑞瑞虎3", "奇瑞瑞虎3X",
                               "奇瑞艾瑞泽M7", "奇瑞风云2", "奔腾B30_2016款", "奔腾B50", "奔腾B50_2011款", "奔腾B50_2012款", "奔腾B50_2013款",
                               "奔腾B70", "奔腾B70_2010款", "奔腾B70_2012款", "奔腾B70_2014款", "奔腾B90", "奔腾X40", "奔腾X80",
                               "奔腾X80_2013款", "奔腾X80_2015款", "奔腾X80_2016款", "奔驰A级", "奔驰A级(进口)", "奔驰A级AMG", "奔驰B级",
                               "奔驰B级_2009款", "奔驰B级_2012款", "奔驰B级_2015款", "奔驰CLA级", "奔驰CLA级AMG", "奔驰CLA级_2016款",
                               "奔驰CLA级_2017款", "奔驰CLK级", "奔驰CLS级", "奔驰CLS级AMG", "奔驰C级", "奔驰C级AMG", "奔驰C级AMG_2012款",
                               "奔驰C级_2010款", "奔驰C级_2011款", "奔驰C级_2013款", "奔驰C级_2015款", "奔驰C级_2016款", "奔驰C级_2017款",
                               "奔驰C级_2018款", "奔驰E级", "奔驰E级_2010款", "奔驰E级_2011款", "奔驰E级_2012款", "奔驰E级_2013款",
                               "奔驰E级_2014款", "奔驰E级_2015款", "奔驰E级_2016款", "奔驰GLA级", "奔驰GLA级_2015款", "奔驰GLA级_2016款",
                               "奔驰GLC级", "奔驰GLC级_2016款", "奔驰GLE级", "奔驰GLE级AMG", "奔驰GLK级", "奔驰GLK级_2011款",
                               "奔驰GLK级_2012款", "奔驰GLK级_2013款", "奔驰GLK级_2014款", "奔驰GLK级_2015款", "奔驰GLS级", "奔驰GL级",
                               "奔驰G级", "奔驰M级", "奔驰R级", "奔驰R级_2010款", "奔驰SLK级", "奔驰S级", "奔驰S级AMG", "奔驰S级_2012款",
                               "奔驰S级_2014款", "奔驰V级", "奔驰_唯雅诺", "奔驰_威霆", "奥迪A1", "奥迪A1_2012款", "奥迪A3", "奥迪A3_2012款",
                               "奥迪A3_2014款", "奥迪A3_2016款", "奥迪A3_2017款", "奥迪A4", "奥迪A4L", "奥迪A4L_2010款", "奥迪A4L_2011款",
                               "奥迪A4L_2012款", "奥迪A4L_2013款", "奥迪A4L_2015款", "奥迪A4L_2016款", "奥迪A4L_2018款", "奥迪A5",
                               "奥迪A6", "奥迪A6L", "奥迪A6L_2010款", "奥迪A6L_2011款", "奥迪A6L_2012款", "奥迪A6L_2015款",
                               "奥迪A6L_2016款", "奥迪A6L_2017款", "奥迪A7", "奥迪A8L", "奥迪A8L_2013款", "奥迪Q3", "奥迪Q3_2012款",
                               "奥迪Q3_2013款", "奥迪Q3_2015款", "奥迪Q3_2016款", "奥迪Q5", "奥迪Q5_2010款", "奥迪Q5_2011款",
                               "奥迪Q5_2012款", "奥迪Q5_2013款", "奥迪Q5_2015款", "奥迪Q5_2016款", "奥迪Q5_2017款", "奥迪Q7",
                               "奥迪Q7_2012款", "奥迪S3", "奥迪S5", "奥迪TT", "奥迪TTS", "威麟V5", "宝沃汽车_宝沃BX5", "宝沃汽车_宝沃BX7",
                               "宝沃汽车宝沃BX7", "宝马1系", "宝马1系_2008款", "宝马1系_2012款", "宝马1系_2013款", "宝马1系_2017款", "宝马2系",
                               "宝马2系旅行车", "宝马3系", "宝马3系GT", "宝马3系_2010款", "宝马3系_2012款", "宝马3系_2013款", "宝马3系_2014款",
                               "宝马3系_2015款", "宝马3系_2016款", "宝马3系_2017款", "宝马3系_2018款", "宝马4系", "宝马5系", "宝马5系GT",
                               "宝马5系_2011款", "宝马5系_2012款", "宝马5系_2013款", "宝马5系_2014款", "宝马5系_2017款", "宝马6系", "宝马7系",
                               "宝马7系_2009款", "宝马7系_2011款", "宝马7系_2013款", "宝马M系", "宝马X1_2010款", "宝马X1_2012款",
                               "宝马X1_2013款", "宝马X1_2014款", "宝马X1_2015款", "宝马X1_2016款", "宝马X1_2018款", "宝马X3",
                               "宝马X3_2012款", "宝马X3_2013款", "宝马X3_2014款", "宝马X4", "宝马X5", "宝马X5_2011款", "宝马X5_2013款",
                               "宝马X5_2014款", "宝马X6", "宝马Z4", "宝骏310", "宝骏310W", "宝骏510", "宝骏510_2017款", "宝骏530",
                               "宝骏560", "宝骏560_2016款", "宝骏560_2017款", "宝骏610", "宝骏630", "宝骏630_2013款", "宝骏730_2014款",
                               "宝骏730_2015款", "宝骏730_2016款", "宝骏730_2017款", "宝骏_乐驰", "川汽_野马F10", "川汽_野马F16", "川汽_野马T70",
                               "川汽_野马T70新能源", "川汽_野马T80", "广汽吉奥GX6", "广汽吉奥_奥轩GX5", "广汽吉奥_星朗", "康迪", "开瑞K50", "开瑞K60EV",
                               "开瑞_优胜2代", "思铭", "捷豹F-PACE", "捷豹XE", "捷豹XF", "捷豹XFL", "捷豹XJ", "捷豹XJ_2016款", "斯威X7",
                               "斯巴鲁BRZ", "斯巴鲁XV", "斯巴鲁XV_2012款", "斯巴鲁_傲虎", "斯巴鲁_力狮", "斯巴鲁_森林人", "斯巴鲁_森林人_2011款",
                               "斯巴鲁_森林人_2012款", "斯巴鲁_森林人_2013款", "斯柯达Yeti", "斯柯达Yeti_2014款", "斯柯达_昊锐", "斯柯达_昊锐_2009款",
                               "斯柯达_昊锐_2012款", "斯柯达_明锐", "斯柯达_明锐_2009款", "斯柯达_明锐_2010款", "斯柯达_明锐_2012款", "斯柯达_明锐_2013款",
                               "斯柯达_明锐_2014款", "斯柯达_明锐_2015款", "斯柯达_明锐_2016款", "斯柯达_昕动_2014款", "斯柯达_昕动_2016款", "斯柯达_昕锐",
                               "斯柯达_昕锐_2013款", "斯柯达_昕锐_2016款", "斯柯达_晶锐", "斯柯达_晶锐_2011款", "斯柯达_晶锐_2012款", "斯柯达_晶锐_2014款",
                               "斯柯达_柯珞克", "斯柯达_柯米克", "斯柯达_柯迪亚克_2018款", "斯柯达_速派", "斯柯达_速派_2013款", "斯柯达_速派_2016款",
                               "斯柯达昕动", "斯达泰克-卫士", "日产", "日产GT-R", "日产NV200", "日产_劲客", "日产_天籁", "日产_天籁_2008款",
                               "日产_天籁_2011款", "日产_天籁_2013款", "日产_天籁_2016款", "日产_奇骏", "日产_奇骏_2010款", "日产_奇骏_2012款",
                               "日产_奇骏_2014款", "日产_奇骏_2015款", "日产_奇骏_2017款", "日产_帕拉丁", "日产_楼兰", "日产_玛驰_2010款",
                               "日产_蓝鸟_2016款", "日产_轩逸", "日产_轩逸_2009款", "日产_轩逸_2012款", "日产_轩逸_2014款", "日产_轩逸_2016款",
                               "日产_轩逸_2018款", "日产_逍客", "日产_逍客_2011款", "日产_逍客_2012款", "日产_逍客_2015款", "日产_逍客_2016款",
                               "日产_逍客_2017款", "日产_途乐", "日产_阳光", "日产_阳光_2011款", "日产_阳光_2014款", "日产_阳光_2015款", "日产_颐达",
                               "日产_颐达_2008款", "日产_骊威", "日产_骊威_2010款", "日产_骊威_2013款", "日产_骊威_2015款", "日产_骏逸", "日产_骐达",
                               "日产_骐达_2008款", "日产_骐达_2011款", "日产_骐达_2014款", "昌河M50", "昌河M70", "昌河Q35", "昌河_福瑞达",
                               "本田CR-V", "本田CR-V_2010款", "本田CR-V_2012款", "本田CR-V_2013款", "本田CR-V_2015款", "本田CR-V_2016款",
                               "本田UR-V_2017款", "本田XR-V_2015款", "本田XR-V_2017款", "本田_冠道_2017款", "本田_凌派", "本田_凌派_2013款",
                               "本田_凌派_2015款", "本田_凌派_2016款", "本田_哥瑞_2016款", "本田_奥德赛", "本田_奥德赛_2015款", "本田_思域",
                               "本田_思域_2009款", "本田_思域_2012款", "本田_思域_2016款", "本田_思铂睿", "本田_思铂睿_2009款", "本田_思铂睿_2015款",
                               "本田_杰德", "本田_杰德_2013款", "本田_杰德_2014款", "本田_杰德_2016款", "本田_歌诗图", "本田_歌诗图_2012款",
                               "本田_歌诗图_2014款", "本田_竞瑞", "本田_缤智", "本田_缤智_2015款", "本田_艾力绅", "本田_艾力绅_2015款",
                               "本田_艾力绅_2016款", "本田_锋范_2015款", "本田_锋范_2017款", "本田_锋范经典", "本田_锋范经典_2008款",
                               "本田_锋范经典_2012款", "本田_雅阁", "本田_雅阁_2010款", "本田_雅阁_2011款", "本田_雅阁_2012款", "本田_雅阁_2013款",
                               "本田_雅阁_2014款", "本田_雅阁_2015款", "本田_雅阁_2016款", "本田_飞度", "本田_飞度_2011款", "本田_飞度_2014款",
                               "本田_飞度_2016款", "本田_飞度_2018款", "林肯MKC", "林肯MKT", "林肯MKX", "林肯MKZ", "林肯_领航员", "林肯大陆",
                               "标致2008", "标致2008_2014款", "标致207", "标致207_2011款", "标致3008", "标致3008_2013款",
                               "标致3008_2015款", "标致301", "标致301_2014款", "标致301_2016款", "标致307", "标致307_2010款",
                               "标致307_2012款", "标致308", "标致308S_2015款", "标致308_2012款", "标致308_2013款", "标致308_2014款",
                               "标致308_2016款", "标致308_SW", "标致4008", "标致407", "标致408", "标致408_2011款", "标致408_2013款",
                               "标致408_2014款", "标致408_2015款", "标致408_2016款", "标致5008", "标致508_2011款", "标致508_2012款",
                               "标致508_2013款", "标致508_2015款", "欧宝_安德拉", "欧宝_雅特", "欧宝安德拉", "欧朗", "比亚迪F0", "比亚迪F3",
                               "比亚迪F3R", "比亚迪F3_2011款", "比亚迪F3_2012款", "比亚迪F3_2013款", "比亚迪F3_2015款", "比亚迪F3_2016款",
                               "比亚迪F6", "比亚迪F6_2010款", "比亚迪G3", "比亚迪G3R", "比亚迪G5_2014款", "比亚迪G6", "比亚迪L3",
                               "比亚迪L3_2012款", "比亚迪M6", "比亚迪S6_2011款", "比亚迪S6_2012款", "比亚迪S6_2013款", "比亚迪S6_2014款",
                               "比亚迪S7", "比亚迪S7_2015款", "比亚迪S7_2016款", "比亚迪_元", "比亚迪_元_2016款", "比亚迪_唐", "比亚迪_唐_2015款",
                               "比亚迪_宋", "比亚迪_宋MAX", "比亚迪_宋_2016款", "比亚迪_思锐", "比亚迪_秦_2014款", "比亚迪_秦_2015款",
                               "比亚迪_秦_2017款", "比亚迪_速锐", "比亚迪_速锐_2012款", "比亚迪_速锐_2014款", "比亚迪_速锐_2015款", "比亚迪秦",
                               "比速汽车_比速M3", "比速汽车_比速T3", "比速汽车_比速T5", "汇众_伊思坦纳", "汉腾X7", "江淮_同悦", "江淮_同悦RS", "江淮_和悦",
                               "江淮_和悦A30", "江淮_瑞风", "江淮_瑞风M2", "江淮_瑞风M3", "江淮_瑞风M3_2015款", "江淮_瑞风M4", "江淮_瑞风M5",
                               "江淮_瑞风M5_2013款", "江淮_瑞风M6", "江淮_瑞风R3", "江淮_瑞风S2", "江淮_瑞风S2_2015款", "江淮_瑞风S3",
                               "江淮_瑞风S3_2014款", "江淮_瑞风S3_2016款", "江淮_瑞风S5", "江淮_瑞风S5_2013款", "江淮_瑞风S7", "江淮_瑞风_2011款",
                               "江淮_瑞风_2012款", "江淮_瑞鹰", "江淮iEV", "江铃_域虎", "江铃_宝典", "江铃_驭胜S330", "江铃_驭胜S350", "江铃考斯特",
                               "江铃集团新能源_江淮iEV6E", "江铃集团新能源_江铃E200", "江铃集团轻汽_骐铃T100", "江铃集团轻汽_骐铃T3", "沃尔沃C30", "沃尔沃S40",
                               "沃尔沃S60", "沃尔沃S60L", "沃尔沃S60L_2015款", "沃尔沃S60L_2016款", "沃尔沃S60L_2018款", "沃尔沃S80L",
                               "沃尔沃S90", "沃尔沃V40", "沃尔沃V60", "沃尔沃XC60", "沃尔沃XC60_2011款", "沃尔沃XC60_2015款", "沃尔沃XC90",
                               "沃尔沃XC_Classic", "海马M3", "海马M6", "海马M8", "海马S5", "海马S5_2015款", "海马S5_2016款",
                               "海马S5_Young", "海马S5青春版", "海马S7", "海马S7_2013款", "海马V70", "海马_丘比特", "海马_普力马", "海马_欢动",
                               "海马_海福星", "海马_福仕达鸿达", "海马_福美来", "海马_福美来MPV", "海马_福美来_2014款", "海马_福美来_2015款", "海马爱尚",
                               "海马骑士", "特斯拉MODEL_S", "猎豹6481", "猎豹CS10", "猎豹CS10_2015款", "猎豹CS10_2017款", "猎豹CS7",
                               "猎豹CS9", "猎豹Q6", "猎豹_飞腾", "猎豹黑金刚", "玛莎拉蒂Ghibli", "玛莎拉蒂Levante", "玛莎拉蒂_总裁", "玛莎拉蒂总裁",
                               "现代_伊兰特", "现代_伊兰特_2007款", "现代_全新胜达", "现代_全新胜达_2013款", "现代_全新胜达_2015款", "现代_劳恩斯_酷派",
                               "现代_名图_2014款", "现代_名图_2016款", "现代_名图_2017款", "现代_名驭", "现代_悦动", "现代_悦动_2010款",
                               "现代_悦动_2011款", "现代_悦纳_2016款", "现代_新胜达", "现代_朗动_2012款", "现代_朗动_2013款", "现代_朗动_2015款",
                               "现代_朗动_2016款", "现代_格越", "现代_瑞奕", "现代_瑞奕_2014款", "现代_瑞纳", "现代_瑞纳_2010款", "现代_瑞纳_2014款",
                               "现代_瑞纳_2016款", "现代_索纳塔", "现代_索纳塔九", "现代_索纳塔九_2015款", "现代_索纳塔八", "现代_索纳塔八_2011款",
                               "现代_索纳塔八_2013款", "现代_索纳塔八_2014款", "现代_维拉克斯", "现代_途胜", "现代_途胜_2009款", "现代_途胜_2013款",
                               "现代_途胜_2015款", "现代_雅尊", "现代_雅绅特", "现代_领动", "现代_领动_2016款", "现代_领翔", "现代_飞思", "现代i30",
                               "现代ix25", "现代ix25_2015款", "现代ix35", "现代ix35_2010款", "现代ix35_2012款", "现代ix35_2013款",
                               "现代ix35_2015款", "现代全新胜达", "现代名图", "现代悦动_2011款", "现代朗动", "现代飞思", "理念S1", "瑞麒G3", "瑞麒G5",
                               "知豆D2", "福特", "福特Mustang", "福特S-MAX", "福特_全顺", "福特_嘉年华", "福特_嘉年华_2009款", "福特_嘉年华_2011款",
                               "福特_嘉年华_2013款", "福特_探险者", "福特_探险者_2013款", "福特_探险者_2016款", "福特_撼路者", "福特_新世代全顺", "福特_猛禽",
                               "福特_福克斯", "福特_福克斯_2009款", "福特_福克斯_2011款", "福特_福克斯_2012款", "福特_福克斯_2013款", "福特_福克斯_2015款",
                               "福特_福克斯_2017款", "福特_福克斯_2018款", "福特_福睿斯", "福特_福睿斯_2015款", "福特_福睿斯_2017款", "福特_经典全顺",
                               "福特_翼搏", "福特_翼搏_2013款", "福特_翼虎", "福特_翼虎_2013款", "福特_翼虎_2015款", "福特_翼虎_2017款",
                               "福特_致胜_2013款", "福特_蒙迪欧", "福特_蒙迪欧-致胜", "福特_蒙迪欧-致胜_2010款", "福特_蒙迪欧-致胜_2011款",
                               "福特_蒙迪欧_2013款", "福特_蒙迪欧_2017款", "福特_途睿欧", "福特_野马", "福特_野马_2017款", "福特_金牛座",
                               "福特_金牛座_2015款", "福特_锐界", "福特_锐界_2012款", "福特_锐界_2015款", "福特_锐界_2016款", "福特福克斯_2012款",
                               "福特翼搏_2013款", "福特翼虎_2013款", "福特蒙迪欧-致胜", "福特蒙迪欧_2013款", "福田_伽途ix7", "福田_拓陆者", "福田_蒙派克E",
                               "福田_迷迪", "福田_风景", "福迪_探索者6", "福迪_揽福", "红旗H7", "纳智捷_U5_SUV", "纳智捷_优6_SUV",
                               "纳智捷_优6_SUV_2014款", "纳智捷_优6_SUV_2015款", "纳智捷_优6_SUV_2016款", "纳智捷_大7_MPV", "纳智捷_大7_SUV",
                               "纳智捷_大7_SUV_2011款", "纳智捷_大7_SUV_2013款", "纳智捷_大7_SUV_2014款", "纳智捷_纳5", "纳智捷_纳5_2013款",
                               "纳智捷_纳5_2015款", "纳智捷_锐3", "纳智捷优6_SUV", "纳智捷大7_MPV", "纳智捷大7_SUV", "美佳", "英致737", "英致G3",
                               "英致G5", "英菲尼迪ESQ_2014款", "英菲尼迪EX系列", "英菲尼迪FX系列", "英菲尼迪G系列", "英菲尼迪JX35", "英菲尼迪M系列",
                               "英菲尼迪Q50", "英菲尼迪Q50L", "英菲尼迪Q50L_2016款", "英菲尼迪Q70L", "英菲尼迪QX30", "英菲尼迪QX50",
                               "英菲尼迪QX50_2015款", "英菲尼迪QX60", "英菲尼迪QX70", "荣威350_2010款", "荣威350_2011款", "荣威350_2012款",
                               "荣威350_2013款", "荣威350_2014款", "荣威350_2015款", "荣威360", "荣威360_2015款", "荣威360_2017款",
                               "荣威550", "荣威550_2009款", "荣威550_2010款", "荣威550_2012款", "荣威550_2013款", "荣威550_2014款",
                               "荣威750", "荣威950_2012款", "荣威RX3", "荣威RX5_2016款", "荣威RX5_2018款", "荣威RX8", "荣威W5", "荣威e550",
                               "荣威eRX5", "荣威ei6_2017款", "荣威i6", "菲亚特500", "菲亚特_博悦", "菲亚特_致悦", "菲亚特_致悦_2014款", "菲亚特_菲翔",
                               "菲亚特_菲翔_2012款", "菲亚特_菲翔_2015款", "菲亚特_菲跃", "西雅特_西亚特LEON", "观致3", "观致5", "讴歌CDX", "讴歌MDX",
                               "讴歌RDX", "讴歌RL", "讴歌TLX", "讴歌ZDX", "起亚", "起亚K2", "起亚K2_2011款", "起亚K2_2012款",
                               "起亚K2_2015款", "起亚K3", "起亚K3S_2014款", "起亚K3_2013款", "起亚K3_2015款", "起亚K3_2016款", "起亚K4",
                               "起亚K4_2014款", "起亚K5", "起亚K5_2011款", "起亚K5_2012款", "起亚K5_2014款", "起亚K5_2016款", "起亚KX3",
                               "起亚KX3_2015款", "起亚KX5_2016款", "起亚KX7", "起亚KX_CROSS", "起亚_2015款", "起亚_佳乐", "起亚_凯尊",
                               "起亚_新佳乐", "起亚_智跑_2011款", "起亚_智跑_2012款", "起亚_智跑_2014款", "起亚_智跑_2015款", "起亚_智跑_2016款",
                               "起亚_极睿", "起亚_欧菲莱斯", "起亚_焕驰", "起亚_狮跑", "起亚_狮跑_2011款", "起亚_狮跑_2012款", "起亚_狮跑_2013款",
                               "起亚_福瑞迪_2009款", "起亚_福瑞迪_2011款", "起亚_福瑞迪_2012款", "起亚_福瑞迪_2014款", "起亚_秀尔", "起亚_秀尔_2013款",
                               "起亚_索兰托", "起亚_索兰托_2012款", "起亚_索兰托_2013款", "起亚_赛拉图", "起亚_赛拉图_欧风", "起亚_速迈", "起亚_锐欧",
                               "起亚_霸锐", "起亚智跑", "起亚福瑞迪", "起亚索兰托L", "路虎_发现", "路虎_发现_2015款", "路虎_发现神行", "路虎_发现神行_2016款",
                               "路虎_发现神行_2018款", "路虎_揽胜", "路虎_揽胜极光", "路虎_揽胜极光_2014款", "路虎_揽胜极光_2015款", "路虎_揽胜运动版",
                               "路虎_神行者2", "路虎_神行者2_2013款", "道奇_酷威", "道奇_酷威_2013款", "道奇_酷搏", "道奇_锋哲", "金杯750", "金杯S50",
                               "金杯_智尚S30", "金杯_智尚S35", "金杯_海狮", "金杯_海狮X30L", "金杯_海狮_2014款", "金杯_蒂阿兹", "金杯_阁瑞斯", "金龙_凯歌",
                               "金龙_凯特", "金龙_金威", "铃木_凯泽西", "铃木_利亚纳", "铃木_利亚纳A6", "铃木_北斗星", "铃木_北斗星X5", "铃木_吉姆尼",
                               "铃木_启悦_2015款", "铃木_天语_SX4", "铃木_天语_SX4_2009款", "铃木_天语_SX4_2011款", "铃木_天语_SX4_2013款",
                               "铃木_天语_尚悦", "铃木_天语_尚悦_2011款", "铃木_奥拓", "铃木_奥拓_2013款", "铃木_派喜", "铃木_维特拉_2016款", "铃木_羚羊",
                               "铃木_超级维特拉", "铃木_锋驭_2014款", "铃木_雨燕", "铃木_雨燕_2011款", "铃木_雨燕_2013款", "铃木_雨燕_2014款",
                               "铃木_雨燕_2016款", "铃木_骁途", "铃木天语_SX4", "铃木天语_尚悦", "铃木奥拓", "铃木维特拉", "铃木羚羊", "长城C20R",
                               "长城C30", "长城C30_2010款", "长城C50", "长城C50_2013款", "长城M1", "长城M2", "长城M4", "长城M4_2012款",
                               "长城M4_2014款", "长城V80", "长城_炫丽", "长城_精灵", "长城_酷熊", "长城_金迪尔", "长城_风骏5", "长城_风骏6",
                               "长安CS15_2016款", "长安CS35_2012款", "长安CS35_2014款", "长安CS35_2015款", "长安CS35_2016款",
                               "长安CS35_2017款", "长安CS55", "长安CS75", "长安CS75_2014款", "长安CS75_2016款", "长安CS75_2017款",
                               "长安CS95", "长安CX20_2011款", "长安CX20_2014款", "长安_凌轩", "长安_奔奔", "长安_奔奔_2014款", "长安_奔奔_2015款",
                               "长安_奔奔mini", "长安_悦翔", "长安_悦翔V3", "长安_悦翔V5_2012款", "长安_悦翔V7_2015款", "长安_悦翔V7_2016款",
                               "长安_睿骋", "长安_睿骋_2014款", "长安_逸动", "长安_逸动DT", "长安_逸动_2012款", "长安_逸动_2013款", "长安_逸动_2014款",
                               "长安_逸动_2015款", "长安_逸动_2016款", "长安商用", "长安商用V5", "长安商用_欧力威", "长安商用_欧尚", "长安商用_欧尚_2016款",
                               "长安商用_欧诺", "长安商用_欧诺_2014款", "长安商用_睿行", "长安商用_睿行S50", "长安商用_神骐F30", "长安商用_金牛星",
                               "长安商用_金牛星_2011款", "长安商用_长安CX70", "长安商用_长安之星", "长安商用_长安之星2", "长安商用_长安之星3", "长安商用_长安之星7",
                               "长安商用_长安之星S460", "长安商用_长安星光", "长安商用_长安星光4500", "长安商用长安之星2", "长安悦翔V3", "长安欧尚_欧尚A800",
                               "长安欧尚_欧尚X70A", "长安跨越", "陆程", "陆风X2", "陆风X5", "陆风X5_2013款", "陆风X7", "陆风X7_2015款",
                               "陆风X7_2016款", "陆风X8", "陆风X9", "雪佛兰", "雪佛兰Express", "雪佛兰_2015款", "雪佛兰_乐风", "雪佛兰_乐风RV",
                               "雪佛兰_乐风_2010款", "雪佛兰_乐驰", "雪佛兰_乐骋", "雪佛兰_创酷", "雪佛兰_创酷_2014款", "雪佛兰_创酷_2016款", "雪佛兰_探界者",
                               "雪佛兰_景程", "雪佛兰_景程_2010款", "雪佛兰_景程_2013款", "雪佛兰_爱唯欧", "雪佛兰_爱唯欧_2011款", "雪佛兰_爱唯欧_2014款",
                               "雪佛兰_科帕奇", "雪佛兰_科帕奇_2010款", "雪佛兰_科帕奇_2011款", "雪佛兰_科帕奇_2012款", "雪佛兰_科帕奇_2014款",
                               "雪佛兰_科帕奇_2015款", "雪佛兰_科沃兹", "雪佛兰_科沃兹_2016款", "雪佛兰_科鲁兹", "雪佛兰_科鲁兹_2009款", "雪佛兰_科鲁兹_2010款",
                               "雪佛兰_科鲁兹_2011款", "雪佛兰_科鲁兹_2012款", "雪佛兰_科鲁兹_2013款", "雪佛兰_科鲁兹_2015款", "雪佛兰_科鲁兹_2016款",
                               "雪佛兰_科鲁兹_2017款", "雪佛兰_赛欧", "雪佛兰_赛欧_2010款", "雪佛兰_赛欧_2013款", "雪佛兰_迈锐宝", "雪佛兰_迈锐宝XL",
                               "雪佛兰_迈锐宝XL_2016款", "雪佛兰_迈锐宝_2012款", "雪佛兰_迈锐宝_2013款", "雪佛兰_迈锐宝_2014款", "雪佛兰_迈锐宝_2016款",
                               "雪佛兰_迈锐宝_2017款", "雪佛兰科鲁兹_2012款", "雪铁龙C2", "雪铁龙C3-XR", "雪铁龙C3-XR_2015款", "雪铁龙C4L_2013款",
                               "雪铁龙C4L_2014款", "雪铁龙C4L_2015款", "雪铁龙C4_Aircross", "雪铁龙C5", "雪铁龙C5_2011款", "雪铁龙C5_2012款",
                               "雪铁龙C5_2013款", "雪铁龙C5_2014款", "雪铁龙_C4世嘉", "雪铁龙_C4世嘉_2016款", "雪铁龙_世嘉", "雪铁龙_世嘉_2011款",
                               "雪铁龙_世嘉_2012款", "雪铁龙_世嘉_2013款", "雪铁龙_世嘉_2014款", "雪铁龙_世嘉_2016款", "雪铁龙_凯旋", "雪铁龙_毕加索",
                               "雪铁龙_爱丽舍", "雪铁龙_爱丽舍_2014款", "雪铁龙_爱丽舍_2015款", "雪铁龙_爱丽舍_2016款", "雷克萨斯CT", "雷克萨斯CT_2014款",
                               "雷克萨斯ES", "雷克萨斯ES_2013款", "雷克萨斯ES_2015款", "雷克萨斯GS", "雷克萨斯GX", "雷克萨斯IS", "雷克萨斯LX",
                               "雷克萨斯NX", "雷克萨斯NX_2015款", "雷克萨斯RC", "雷克萨斯RX", "雷克萨斯RX经典", "雷克萨斯RX经典_2013款", "雷诺",
                               "雷诺_卡缤", "雷诺_塔利斯曼", "雷诺_科雷傲", "雷诺_科雷傲_2012款", "雷诺_科雷嘉", "雷诺_科雷嘉_2016款", "雷诺_风朗", "雷诺科雷傲",
                               "青年莲花_莲花L3", "青年莲花_莲花L5", "领克01", "风骏房车", "马自达2", "马自达2劲翔", "马自达3", "马自达3_2010款",
                               "马自达3星骋", "马自达3星骋_2011款", "马自达3星骋_2012款", "马自达3星骋_2013款", "马自达5", "马自达6", "马自达6_2008款",
                               "马自达6_2011款", "马自达6_2012款", "马自达6_2013款", "马自达6_2015款", "马自达8", "马自达CX-4",
                               "马自达CX-4_2016款", "马自达CX-5", "马自达CX-5_2013款", "马自达CX-5_2014款", "马自达CX-5_2015款", "马自达CX-7",
                               "马自达_昂克赛拉_2014款", "马自达_昂克赛拉_2016款", "马自达_昂克赛拉_2017款", "马自达_睿翼", "马自达_睿翼_2009款",
                               "马自达_睿翼_2010款", "马自达_睿翼_2012款", "马自达_阿特兹", "马自达_阿特兹_2014款", "马自达_阿特兹_2015款", "马自达昂克赛拉",
                               "马自达阿特兹", "黄海_旗胜V3"};
    for (auto &s : car_list) {
        car_types.push_back(s);
    }
    classNet = readNetFromCaffe(model_path + "/reg.prototxt", model_path + "/reg.bin");

}

pair<string, float> VID::recognize(const Mat &img) {
    auto blob = cv::dnn::blobFromImage(img, 1.0, Size(img.cols * (224.0 / img.rows), 224), cv::Scalar(0.0, 0.0, 0.0),
                                       true, false);
    // DEBUG

    float mean_rgb[3] = {116.779, 103.939, 123.68};
    float std_rgb[3] = {57.12, 57.375, 58.393};
    float scale = 1.0;
    auto *header = (float *) blob.data;
    int size = blob.size[2] * blob.size[3];
    for (int c = 0; c < 3; c++) {
        for (int k = 0; k < size; k++)
            header[c * size + k] = static_cast<float>((header[c * size + k] / scale - mean_rgb[c]) / std_rgb[c]);
    }


    classNet.setInput(blob);
    Mat prob = classNet.forward();
    Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;
    return make_pair(car_types[classId], confidence);
}

VID::VID(char *buffer) {
    rapidjson::Document d;
    d.Parse(buffer);
    auto doc = d["Recognition"].GetObject();
    enabled = doc["VID_enabled"].GetBool();
    if(enabled)
        new(this) VID(std::string(doc["VID_model_path"].GetString()));
}
