# <object_category>    The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1), people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), others(11))

# 0: car, 1: bus, 2: person
from PIL import Image
import os
from shutil import copyfile

type_mapping = {
    '0': -1,
    '1': 2,
    '2': 2,
    '3': 3,
    '4': 0,
    '5': 0,
    '6': 0,
    '7': -1,
    '8': -1,
    '9': 1,
    '10': 3,
    '11': -1
}

in_img_path = 'VisDrone2018-DET-train/images/'
in_annotation_path = 'VisDrone2018-DET-train/annotations/'
out_img_path = 'dataset/images/'
out_annotation_path = 'dataset/labels/'
type_cnt = {}

for img_file in os.listdir(in_img_path):
    filename = img_file.split('.')[0]
    img = Image.open(in_img_path + img_file)
    ori_w, ori_h = img.size
    txt_path = in_annotation_path + filename + '.txt'
    txt_file = open(txt_path, 'r')
    out_file = open(out_annotation_path + filename + '.txt', 'w')
    for line in txt_file.readlines():
        typeid = line.strip().split(',')[-3]
        if type_mapping[typeid] == -1:
            continue
        bbox_left = int(line.strip().split(',')[0])
        bbox_top = int(line.strip().split(',')[1])
        bbox_width = int(line.strip().split(',')[2])
        bbox_height = int(line.strip().split(',')[3])
        h = round(float(bbox_height) / ori_h, 6)
        w = round(float(bbox_width) / ori_w, 6)
        left = round(float(bbox_left) / ori_w, 6)
        top = round(float(bbox_top) / ori_h, 6)
        x = min(round(left + w / 2, 6), 1)
        y = min(round(top + h / 2, 6), 1)
        if str(type_mapping[typeid]) not in type_cnt:
            type_cnt[str(type_mapping[typeid])] = 1
        else:
            type_cnt[str(type_mapping[typeid])] += 1

        out_file.write(str(type_mapping[typeid]) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
    txt_file.close()
    out_file.close()
    copyfile(os.path.join(in_img_path, img_file), os.path.join(out_img_path, img_file))

print(type_cnt)
