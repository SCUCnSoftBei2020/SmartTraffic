from tools.get_yolov5_driveu import DriveuDatabase
import cv2
import os

save_path = './traffic-light'

type_mapping = {
    '0': 'off',
    '1': 'red',
    '2': 'yellow',
    '4': 'green'
}

type_cnt = {k: 0 for k in type_mapping.keys()}
city_list = ['Bochum_all.yml', ]

_filter = lambda class_id: len(class_id) == 6 and \
                           class_id[0] == '1' and \
                           class_id[3] == '3' and \
                           class_id[4] in type_mapping.keys()

for city_yaml in city_list:
    database = DriveuDatabase(city_yaml)
    database.open('./DriveU')
    for img in database.images:
        _, img_mat = img.get_image()
        ori_h, ori_w, _ = img_mat.shape
        result = []  # contains (img_cropped, class_label(str))
        for obj in img.objects:
            if _filter(obj.class_id):
                try:
                    y2 = min(ori_h, obj.y + obj.height)
                    x2 = min(ori_w, obj.x + obj.width)
                    result.append((img_mat[obj.y: y2, obj.x: x2], obj.class_id[4]))
                    type_cnt[obj.class_id[4]] += 1
                except KeyError:
                    continue
        if len(result) > 0:
            for image, label in result:
                light_path = os.path.join(save_path, type_mapping[label])
                if not os.path.exists(light_path):
                    os.mkdir(light_path)
                cv2.imwrite(os.path.join(light_path, img.name + '.jpg'), image)

print(type_cnt)
