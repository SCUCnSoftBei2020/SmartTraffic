import time
from typing import List
import cv2
import torch
import json

from torch.autograd import Variable
from deepsort_wrapper import Deepsort_original
from utils.general import non_max_suppression, scale_coords, xyxy2ltwh, my_xywh2xyxy
from utils.pytorch import select_device
from models.dataloader import LoadImages
from models.experimental import attempt_load
from torchvision import transforms
from deep_sort.cosine_metric_net import TrafficNet
from tqdm import tqdm
import numpy as np
import os


class TrafficLightReg:
    def __init__(self, annotation_path, model_path='./trafficnet.pt'):
        ckpt = torch.load(model_path)
        type_mapping: dict = ckpt['type']  # {'green': 0}
        self._type = {k: v for v, k in type_mapping.items()}  # {0: 'green'}
        self.model = TrafficNet(num_classes=len(type_mapping)).cuda()
        self.model.load_state_dict(ckpt['model_dict'])
        self.model.eval()

        self.traffic_bbox = []
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        with open(annotation_path, 'r') as f:
            first_line = f.readline()
            line_num, stop_line, self.traffic_cnt = list(map(int, first_line.strip().split(' ')[:3]))
            for _ in range(line_num):  # skip line_num
                f.readline()
            for k in range(self.traffic_cnt):
                line = f.readline()
                x1, y1, x2, y2 = tuple(map(int, line.strip().split(' ')))
                self.traffic_bbox.append((x1, y1, x2, y2))

    def predict(self, img: np.ndarray):
        img = self.transforms(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            img = Variable(img).cuda()
            output = self.model(img)
            pred = output.max(dim=0)[1]
            pred = pred.squeeze()
        return self._type[int(pred)]

    def detect(self, image: np.ndarray):
        ret = []
        for bbox in self.traffic_bbox:
            im_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            im_crop = cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB)
            ret.append(self.predict(im_crop))
        return ret


class DeepSORTTracking:
    def __init__(self, config: dict):
        self.config = config
        wt_path = self.config["Track"]["weight_path"]
        self.deepsort = Deepsort_original(wt_path=wt_path)

    def track(self, frame, frm_id, detections, out_scores):
        t0 = time.time()
        frame = frame.astype(np.uint8)
        detections = np.array(detections)
        out_scores = np.array(out_scores)
        tracker, detections_class = self.deepsort.run_deep_sort(frame, out_scores, detections)
        result = []  # contains tuple like `(bbox, id_num)`
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            id_num = str(track.track_id)
            result.append((bbox, id_num))
        print('Tracking at frame %d consumes %.3fs' % (frm_id, time.time() - t0))
        return result


class TorchDetection:
    track_ignore_list = ('person', 'traffic-red', 'traffic-yellow', 'traffic-green', 'motor')
    color_mapping = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'off': (0, 0, 0)
    }
    mapping_to_render = {
        'off': 0,
        'red': 1,
        'green': 2,
        'yellow': 3
    }

    def __init__(self, json_path='./config-dalukou.json', model_path='yolov5l-last.pt', device='', in_det_file=None):
        # params here will be overwritten by config.json if not left empty in json file
        self.config = json.load(open(json_path, 'r'))
        self.model_path = self.config['Detection']['model_path'] \
            if 'model_path' in self.config['Detection'] else model_path
        self.vdo_path = self.config['Detection']['vdo_path']
        self.out_det_path = self.config['Detection']['out_det_path']
        self.mask_path = self.config['Detection']['mask_path']
        if self.mask_path:
            self.mask = self.get_mask(self.mask_path)
            self.mask = np.expand_dims(self.mask, 2)
            self.mask = np.repeat(self.mask, 3, 2)
        else:
            self.mask = None
        self.use_device = select_device(device)
        self.half = self.use_device.type != 'cpu'
        self.in_det_file = in_det_file
        if self.in_det_file is None:
            self.model = attempt_load(model_path, map_location=self.use_device)
            if self.half:
                self.model.half()
        self.track = DeepSORTTracking(self.config)
        self.traffic_light_reg = TrafficLightReg(self.config['Preparation']['out_annotation'])
        self.traffic_path = self.config['Detection']['traffic_path']

    def detect_alone(self, vdo_path=None, imgsz=640, augment=False, conf_thres=0.4, iou_thres=0.5, classes=None,
                     agnostic_nms=False, out_det_path=None):
        if vdo_path is None:
            vdo_path = self.vdo_path
        assert out_det_path is not None
        dataset = LoadImages(vdo_path, img_size=imgsz, mask=self.mask)
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        det_file = open(out_det_path, 'w')
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.use_device)  # init img
        _ = self.model(img.half() if self.half else img) if self.use_device.type != 'cpu' else None

        for _, img, img0, _, frm_id in tqdm(dataset):  # img: RGB, img0: BGR
            img = torch.from_numpy(img).to(self.use_device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time.time()
            pred: List[torch.Tensor] = self.model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes,
                                       agnostic_nms)
            t2 = time.time()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0 = '', img0.shape
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    for *xyxy, conf, cls in det:  # for each frame
                        ltwh: list = (xyxy2ltwh(torch.tensor(xyxy).view(1, 4))).view(
                            -1).tolist()  # normalized xywh
                        print("%d,%d,%d,%d,%d,%d,%.3f,1,-1,-1,%s" %
                              (frm_id, -1, *ltwh, conf, names[int(cls)]), file=det_file)
                print('%sDetection Done in %.3fs' % (s, t2 - t1))

        print('Video Done in %.3fs' % (time.time() - t0))
        det_file.close()

    def detect(self, vdo_path=None, imgsz=640, augment=False, conf_thres=0.4, iou_thres=0.5, classes=None,
               agnostic_nms=False, progress_file='./progress.app', visualize=False):
        if os.path.exists(self.out_det_path):
            print('Using existing tracking files...')
            with open(progress_file, 'w') as f:
                print("%d/%d/%d" % (1, 1, 1), file=f)
            return
        if vdo_path is None:
            vdo_path = self.vdo_path
        txt_path = self.out_det_path
        txt_file = open(txt_path, 'w')
        dataset = LoadImages(vdo_path, img_size=imgsz, mask=self.mask)
        if self.in_det_file is None:
            img = torch.zeros((1, 3, imgsz, imgsz), device=self.use_device)  # init img
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            _ = self.model(img.half() if self.half else img) if self.use_device.type != 'cpu' else None  # run once

        t0 = time.time()

        traffic_list = []  # [(frm_id, light_type(indicated in mapping_to_render))]
        if self.in_det_file is not None:
            det_dict = {frm: [] for frm in range(1, dataset.nframes + 1)}
            with open(self.in_det_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip().split(',')  # [frm_id, -1, *ltwh, conf, names[int(cls)]
                    det_dict[int(line[0])].append([*line[2:7], line[10]])
        for _, img, img0, _, frm_id in tqdm(dataset):  # img: RGB, img0: BGR
            img = torch.from_numpy(img).to(self.use_device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # process traffic light
            t_traffic = time.time()
            traffic_result = self.traffic_light_reg.detect(img0)  # ['red', 'red']
            traffic_obj = [frm_id, ]
            traffic_obj.extend([self.mapping_to_render[ret] for ret in traffic_result])
            traffic_list.append(traffic_obj)  # [frm_id, traffic light1, traffic light2]
            if visualize:
                for idx, bbox in enumerate(self.traffic_light_reg.traffic_bbox):
                    cv2.rectangle(img0, bbox[0:2], bbox[2:4], self.color_mapping[traffic_result[idx]])
            print('Traffic Done in %.3fs' % (time.time() - t_traffic))

            # finished traffic light
            if self.in_det_file is None:
                # Inference
                t1 = time.time()
                pred: List[torch.Tensor] = self.model(img, augment=augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes,
                                           agnostic_nms)
                t2 = time.time()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    s, im0 = '', img0.shape
                    s += '%gx%g ' % img.shape[2:]  # print string
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        # Write results
                        detections, out_scores = [], []
                        for *xyxy, conf, cls in det:  # for each frame
                            ltwh: list = (xyxy2ltwh(torch.tensor(xyxy).view(1, 4))).view(
                                -1).tolist()  # normalized xywh
                            if names[int(cls)] not in self.track_ignore_list:
                                detections.append(ltwh)
                                out_scores.append(conf)
                            else:  # ignored, do not track, set obj_id to -1
                                print("%d,%d,%d,%d,%d,%d,%.3f,1,-1,-1,%s" %
                                      (frm_id, -1, *ltwh, conf, names[int(cls)]), file=txt_file)
                                if visualize:
                                    xyxy = tuple(map(lambda tensor: int(tensor), xyxy))
                                    cv2.rectangle(img0, xyxy[0:2], xyxy[2:4], (255, 255, 255), 2)
                                    cv2.putText(img0, names[int(cls)], (int(ltwh[0]), int(ltwh[1])), 0, 5e-3 * 200,
                                                (0, 255, 0), 2)
                        if detections:  # if any detections exist
                            ret = self.track.track(img0, frm_id, detections, out_scores)
                            for track_ret in ret:
                                # track_ret = (bbox, id_num)
                                if visualize:
                                    xyxy_2 = tuple(map(lambda tensor: int(tensor), track_ret[0][0]))
                                    cv2.rectangle(img0, xyxy_2[0:2], xyxy_2[2:4], (255, 255, 255), 2)
                                    cv2.putText(img0, str(track_ret[1]),
                                                (int(track_ret[0][0][0]), int(track_ret[0][0][1])),
                                                0, 5e-3 * 200, (0, 255, 0), 2)

                                print("%d,%d,%d,%d,%d,%d,%.3f,1,-1,-1,-1" %
                                      (frm_id, int(track_ret[1]), int(track_ret[0][0][0]), int(track_ret[0][0][1]),
                                       int(track_ret[0][0][2] - track_ret[0][0][0]),
                                       int(track_ret[0][0][3] - track_ret[0][0][1]), track_ret[0][1]),
                                      file=txt_file)

                        with open(progress_file, 'w') as f:
                            print("%d/%d/%d" % (1, frm_id, dataset.nframes), file=f)

                    print('%sDetection Done in %.3fs' % (s, t2 - t1))
            else:  # load det_dict
                t1 = time.time()
                detections, out_scores = [], []
                for obj in det_dict[frm_id]:  # for each object in this frame
                    # obj: [ltwh, conf, name]
                    ltwh, conf, name = obj[0:4], float(obj[4]), obj[5]
                    ltwh = list(map(int, ltwh))
                    if name not in self.track_ignore_list:
                        detections.append(ltwh)
                        out_scores.append(conf)
                    else:
                        print("%d,%d,%d,%d,%d,%d,%.3f,1,-1,-1,%s" %
                              (frm_id, -1, *ltwh, conf, name), file=txt_file)
                        if visualize:
                            xyxy = my_xywh2xyxy(torch.tensor(ltwh).view(1, 4)).view(-1).tolist()
                            xyxy = tuple(map(lambda tensor: int(tensor), xyxy))
                            cv2.rectangle(img0, xyxy[0:2], xyxy[2:4], (255, 255, 255), 2)
                            cv2.putText(img0, name, (int(ltwh[0]), int(ltwh[1])), 0, 5e-3 * 200,
                                        (0, 255, 0), 2)
                if detections:  # if any detections exist
                    ret = self.track.track(img0, frm_id, detections, out_scores)
                    for track_ret in ret:
                        # track_ret = (bbox, id_num)
                        if visualize:
                            xyxy_2 = tuple(map(lambda tensor: int(tensor), track_ret[0][0]))
                            cv2.rectangle(img0, xyxy_2[0:2], xyxy_2[2:4], (255, 255, 255), 2)
                            cv2.putText(img0, str(track_ret[1]), (int(track_ret[0][0][0]), int(track_ret[0][0][1])),
                                        0, 5e-3 * 200, (0, 255, 0), 2)

                        print("%d,%d,%d,%d,%d,%d,%.3f,1,-1,-1,-1" %
                              (frm_id, int(track_ret[1]), int(track_ret[0][0][0]), int(track_ret[0][0][1]),
                               int(track_ret[0][0][2] - track_ret[0][0][0]),
                               int(track_ret[0][0][3] - track_ret[0][0][1]), track_ret[0][1]),
                              file=txt_file)

            if visualize:
                fps_ed_time = time.time()
                fps = 1 / (fps_ed_time - t1)
                cv2.putText(img0, 'FPS: %.1f' % fps, (10, 40), 0, 5e-3 * 200, (0, 255, 0), 2)
                cv2.imshow('visualize', img0)
                cv2.waitKey(1)

        with open(self.traffic_path, 'w') as f:
            format_str = ' '.join(['%s'] * (len(self.traffic_light_reg.traffic_bbox) + 1))
            f.write('\n'.join([format_str % tuple(obj) for obj in traffic_list]))
        print('Video Done in %.3fs' % (time.time() - t0))
        txt_file.close()
        cv2.destroyAllWindows()

    def get_mask(self, path: str):
        assert os.path.exists(path)
        mask = cv2.imread(path, 0)
        mask = mask / 255.0
        return mask


class Speed:
    def __init__(self, json_path):
        self.json_path = json_path

    def calSpeed(self):
        os.system('./retest %s' % self.json_path)


if __name__ == '__main__':
    det = TorchDetection(json_path='./config-dalukou.json')
    # det.detect()
    det.detect(visualize=True)
    # det.detect_alone(out_det_path='./det/det-chusai.txt')
