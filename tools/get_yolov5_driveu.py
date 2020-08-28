import os

import cv2
import numpy as np
import yaml


class DriveuObject:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.class_id = 0

    def parse_object_dict(self, object_dict: dict):
        self.x = object_dict["x"]
        self.y = object_dict["y"]
        self.width = object_dict["width"]
        self.height = object_dict["height"]
        self.class_id = str(object_dict["class_id"])


class DriveuImage:
    def __init__(self):
        self.img_name = ""
        self.file_path = ""
        self.disp_file_path = ""
        self.timestamp = 0
        self.objects = []

    def parse_image_dict(self, image_dict: dict, data_base_dir: str = ""):
        # Parse images
        if data_base_dir != "":
            inds = [i for i, c in enumerate(image_dict["path"]) if c == "/"]
            self.file_path = (
                    data_base_dir + "/" + image_dict["path"][inds[-4]:]
            )
            inds = [
                i for i, c in enumerate(image_dict["disp_path"]) if c == "/"
            ]
            self.disp_file_path = (
                    data_base_dir + "/" + image_dict["disp_path"][inds[-4]:]
            )
        else:
            self.file_path = image_dict["path"]
            self.disp_file_path = image_dict["disp_path"]
            self.timestamp = image_dict["time_stamp"]

        # Parse labels
        for o in image_dict["objects"]:
            label = DriveuObject()
            label.parse_object_dict(o)
            self.objects.append(label)

    def get_image(self):
        if os.path.isfile(self.file_path):
            self.name = os.path.splitext(os.path.basename(self.file_path))[0]
            # Load image from file path, do debayering and shift
            img = cv2.imread(self.file_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)
            # Images are saved in 12 bit raw -> shift 4 bits
            img = np.right_shift(img, 4)
            img = img.astype(np.uint8)

            return True, img
        else:
            raise FileNotFoundError

    def get_bbox(self):
        return [[o.x, o.y, o.width, o.height, o.class_id] for o in self.objects]


class DriveuDatabase:
    def __init__(self, file_path):
        self.images = []
        self.file_path = file_path

    def open(self, data_base_dir: str = ""):
        if os.path.exists(self.file_path):
            images = yaml.load(open(self.file_path, "rb").read())
        else:
            raise FileNotFoundError
        for image_dict in images:
            # parse and store image
            image = DriveuImage()
            image.parse_image_dict(image_dict, data_base_dir)
            self.images.append(image)


if __name__ == '__main__':
    out_img_path = 'dataset/images/'
    out_annotation_path = 'dataset/labels/'
    # 4: red, 5: yellow 6: green
    type_mapping = {
        '1': '4',  # red
        '2': '5',  # yellow
        '4': '6'  # green
    }
    type_counter = {
        '1': 0,
        '2': 0,
        '4': 0
    }
    _filter = lambda class_id: len(class_id) == 6 and \
                               class_id[0] == '1' and \
                               class_id[3] == '3'
    city_list = ['Bochum_all.yml', 'Bremen_all.yml']
    for city_yaml in city_list:
        database = DriveuDatabase(city_yaml)
        database.open('./DriveU')
        for idx, img in enumerate(database.images):  # type: img: DriveuImage
            result_list = []
            _, img_mat = img.get_image()
            ori_h, ori_w, _ = img_mat.shape
            for obj in img.objects:
                if _filter(obj.class_id):
                    h = round(float(obj.height) / ori_h, 6)
                    w = round(float(obj.width) / ori_w, 6)
                    left = round(float(obj.x) / ori_w, 6)
                    top = round(float(obj.y) / ori_h, 6)
                    x = min(round(left + w / 2, 6), 1)
                    y = min(round(top + h / 2, 6), 1)
                    x = max(x, 0)
                    y = max(y, 0)
                    try:
                        result_list.append(type_mapping[obj.class_id[4]] + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))
                        type_counter[obj.class_id[4]] += 1
                    except KeyError:  # other traffic light
                        continue
            if len(result_list) > 0:
                # if os.path.exists(os.path.join(out_img_path, img.name) + '.jpg'):
                #     raise FileExistsError
                cv2.imwrite(os.path.join(out_img_path, img.name) + '.jpg', img_mat)
                txt_path = os.path.join(out_annotation_path, img.name) + '.txt'
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(result_list))
    print(type_counter)