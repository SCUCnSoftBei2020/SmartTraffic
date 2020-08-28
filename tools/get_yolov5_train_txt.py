import os
import random

img_path = 'dataset/images/'
label_path = 'dataset/labels/'
train_val_ratio = 0.9  #


def check_exist_txt(jpg_name: str):
    txt_path = os.path.join(label_path, jpg_name.split('.')[-2]) + '.txt'
    if os.path.exists(txt_path):
        return True
    else:
        print(txt_path)
        return False


file_list = [jpg_file for jpg_file in os.listdir(img_path)
             if jpg_file.split('.')[-1] == 'jpg']

indices = list(range(len(file_list)))
random.shuffle(indices)

train_indices = indices[:int(len(file_list) * train_val_ratio)]
val_indices = indices[int(len(file_list) * train_val_ratio) + 1:]

f = open('train.txt', 'w')
f2 = open('test.txt', 'w')

for train_idx in train_indices:
    if not check_exist_txt(file_list[train_idx]):
        continue
    print(os.path.abspath(img_path + file_list[train_idx]), file=f)

for val_idx in val_indices:
    if not check_exist_txt(file_list[val_idx]):
        continue
    print(os.path.abspath(img_path + file_list[val_idx]), file=f2)

f.close()
f2.close()
