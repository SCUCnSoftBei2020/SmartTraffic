import xml.etree.ElementTree as ET
import os
from shutil import copyfile

path, _ = os.path.split(os.path.realpath(__file__))
print(path)

out_path = '/dataset/images/'
out_labels_path = '/dataset/labels/'

img_path = '/img/'
train_xml_path = "/DETRAC-Train-Annotations-XML-v3/"
val_xml_path = "/DETRAC-Test-Annotations-XML/"

frame_gap = 20

all_type = {'Truck-Pickup': -1, 'Bus': 1, 'Taxi': 0, 'Van': 0, 'Truck-Box-Large': -1, 'Police': 0, 'MiniVan': 0,
            'Hatchback': 0, 'Sedan': 0, 'Suv': 0, 'Truck-Flatbed': -1, 'Truck-Box-Med': -1, 'Truck-Util': -1,
            'car': 0, 'bus': 1, 'van': 0, 'others': -1}
# 0: car, 1: bus, 2: person

# type_cnt = {}

# final_cnt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

'''
//////////////////////////////////
        notation files
//////////////////////////////////
'''


def get_txt_label(xml_path: str, frame_gap: int):  # both train and val go into out_path
    '''
    xml_path:
    frame_gap:
    file_handler:
    '''
    type_cnt = {}
    final_cnt = {0: 0, 1: 0, 2: 0}

    not_file_list = os.listdir(path + xml_path)

    for xmlfile in not_file_list:
        if xmlfile[-3:] != "xml":
            continue
        filename = path +xml_path + "/" + xmlfile
        # print(filename)

        tree = ET.parse(filename)
        root = tree.getroot()
        file_id = root.attrib['name']
        all_frame = root.findall('frame')
#         for i in range(len(all_frame)):
#           assert i == int(all_frame[i].attrib['num']) - 1
        for idx in range(0, len(all_frame), frame_gap):
            child = all_frame[idx]
            frame_id = file_id + "-" + str(int(child.attrib['num']) - 1)
            outfile = frame_id + ".txt"
            # print(outfile)
            f = open(path + "/" + out_labels_path + outfile, "w")
            child = child.find("target_list")

            for target in child.findall('target'):
                _box = target.find('box').attrib
                _type = target.find("attribute").attrib['vehicle_type']
                if _type not in type_cnt:
                    type_cnt[_type] = 1
                else:
                    type_cnt[_type] += 1

                if all_type[_type] == -1:
                    continue
                final_cnt[all_type[_type]] += 1

                _type = str(all_type[_type])
                top = round(float(_box['top']) / 540, 6)
                left = round(float(_box['left']) / 960, 6)
                h = round(float(_box['height']) / 540, 6)
                w = round(float(_box['width']) / 960, 6)
                x = min(round(left + w / 2, 6), 1)
                y = min(round(top + h / 2, 6), 1)
                _info = _type + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
                f.write(_info)
                # print(frame_id,_info)
            f.close()
    print(type_cnt)
    print(final_cnt)


if __name__ == '__main__':

    get_txt_label(train_xml_path, frame_gap)
    get_txt_label(val_xml_path, frame_gap)
    for txt_file in os.listdir(path + out_labels_path):
        if txt_file.split('.')[-1] != 'txt':
            continue
        seq_name = txt_file.split('-')[0]  # 'MVI_20021'
        frame_id = txt_file.split('-')[1].split('.')[0]  # '40'
        target_img_name = 'img%05d.jpg' % (int(frame_id) + 1)
        copyfile(path + img_path + seq_name + '/' + target_img_name, path + out_path + txt_file.split('.')[0] + '.jpg')
