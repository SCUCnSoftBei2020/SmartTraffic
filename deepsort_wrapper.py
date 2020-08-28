import collections

from deep_sort.cosine_metric_net import CosineMetricNet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection

import numpy as np

import torch
import torchvision


class Deepsort_original(object):
    def __init__(self, wt_path='ckpts/cos_metric_net.pt', wt_type='cosine'):
        if wt_type == 'cosine':
            exclude_keys = ['weights', 'scale']  # these params useless in eval stage
            self.encoder = CosineMetricNet(num_classes=-1, add_logits=False).cuda()
            try:
                ckpt:collections.OrderedDict = torch.load(wt_path)
                #  type: ckpt['model_state_dict']: dict
                ckpt['model_state_dict'] = {k: v for k, v in ckpt['model_state_dict'].items()
                                            if k not in exclude_keys}
                self.encoder.load_state_dict(ckpt['model_state_dict'], strict=False)
            except KeyError as e:
                s = "Model loaded(%s) is not compatible with the definition, please check!" % wt_path
                raise KeyError(s) from e

        elif wt_type == 'siamese':
            self.encoder = torch.load(wt_path)
        else:
            raise NotImplementedError

        self.encoder = self.encoder.eval()
        print("Deep sort model loaded")

        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", .5, 100)
        self.tracker = Tracker(self.metric)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def reset_tracker(self):
        self.tracker = Tracker(self.metric)

    # Deep sort needs the format `top_left_x, top_left_y, width,height

    def format_yolo_output(self, out_boxes):
        for b in range(len(out_boxes)):
            out_boxes[b][0] = out_boxes[b][0] - out_boxes[b][2] / 2
            out_boxes[b][1] = out_boxes[b][1] - out_boxes[b][3] / 2
        return out_boxes

    def pre_process(self, frame, detections):

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 64)),
            torchvision.transforms.ToTensor(),
        ])

        crops = []
        for d in detections:

            for i in range(len(d)):
                if d[i] < 0:
                    d[i] = 0

            img_h, img_w, img_ch = frame.shape

            xmin, ymin, w, h = d

            if xmin > img_w:
                xmin = img_w

            if ymin > img_h:
                ymin = img_h

            xmax = xmin + w
            ymax = ymin + h

            ymin = abs(int(ymin))
            ymax = abs(int(ymax))
            xmin = abs(int(xmin))
            xmax = abs(int(xmax))

            try:
                crop = frame[ymin:ymax, xmin:xmax, :]
                crop = transforms(crop)
                crops.append(crop)
            except:
                continue

        crops = torch.stack(crops)

        return crops

    def extract_features_only(self, frame, coords):

        for i in range(len(coords)):
            if coords[i] < 0:
                coords[i] = 0

        img_h, img_w, img_ch = frame.shape

        xmin, ymin, w, h = coords

        if xmin > img_w:
            xmin = img_w

        if ymin > img_h:
            ymin = img_h

        xmax = xmin + w
        ymax = ymin + h

        ymin = abs(int(ymin))
        ymax = abs(int(ymax))
        xmin = abs(int(xmin))
        xmax = abs(int(xmax))

        crop = frame[ymin:ymax, xmin:xmax, :]
        # crop = crop.astype(np.uint8)

        # print(crop.shape,[xmin,ymin,xmax,ymax],frame.shape)

        crop = self.transforms(crop)
        crop = crop.cuda()

        gaussian_mask = self.gaussian_mask

        input_ = crop * gaussian_mask
        input_ = torch.unsqueeze(input_, 0)

        features = self.encoder.forward_once(input_)
        features = features.detach().cpu().numpy()

        corrected_crop = [xmin, ymin, xmax, ymax]

        return features, corrected_crop

    def run_deep_sort(self, frame, out_scores, out_boxes):

        if out_boxes == []:
            self.tracker.predict()
            print('No detections')
            trackers = self.tracker.tracks
            return trackers

        detections = np.array(out_boxes)
        # features = self.encoder(frame, detections.copy())

        processed_crops = self.pre_process(frame, detections).cuda()
        # processed_crops = self.gaussian_mask * processed_crops

        features = self.encoder.forward_once(processed_crops)
        features = features.detach().cpu().numpy()

        if len(features.shape) == 1:
            features = np.expand_dims(features, 0)

        dets = [Detection(bbox, score, feature) \
                for bbox, score, feature in \
                zip(detections, out_scores, features)]

        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker, dets
