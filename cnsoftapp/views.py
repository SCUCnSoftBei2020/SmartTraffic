from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from wrapper import TorchDetection, Speed
import multiprocessing
from multiprocessing import Process
from utils.general import Log, trans_vio
import os
import cv2
import json
import glob


class RequestHandler:
    def __init__(self, config_path):
        multiprocessing.set_start_method('spawn')
        self.config_path = config_path
        self.config_dict = None
        self.load_config()
        self.det_process = None
        self.spd_process = None
        self.config_list = [json_file for json_file in glob.glob('config-*.json')]  # help to switch videos quickly

    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config_dict = json.load(f)

    @staticmethod
    def read_progress():
        stage, curr, all = 1, 0, 1
        try:
            with open("./progress.app", 'r') as f:
                line = f.readline().strip().split("/")
                if len(line) == 3:
                    stage, curr, all = int(line[0]), int(line[1]), int(line[2])
        except FileNotFoundError:
            Log.warn("No such progress file, maybe detection process not loaded yet.")
        return stage, curr, all

    @staticmethod
    def index(request):
        if request.method == 'GET':
            return render(request, 'start.html')

    def result(self, request, config_name='自行上传'):
        if request.method == 'GET':
            config_list = [config_name, ]  # rendered in the first place
            config_list.extend([config for config in self.config_list if config != config_name])
            if config_name != '自行上传' and config_name != self.config_path:  # video changed
                self.spd_process = None
                if os.path.exists('./progress.app'):
                    os.remove('./progress.app')
                self.config_path = config_name
                self.load_config()

                # start detection
                det = TorchDetection(json_path=self.config_path,
                                     in_det_file=self.config_dict["Detection"]["in_det_path"])
                self.det_process = Process(target=det.detect)
                self.det_process.start()
                # init
            return render(request, 'process.html', {'options': config_list})

    @csrf_exempt
    def upload(self, request):
        if request.method == 'POST':
            if os.path.exists('./progress.app'):
                os.remove('./progress.app')
            f = request.FILES['file']
            with open(str(f), 'wb+') as dest:
                for chuck in f.chunks():
                    dest.write(chuck)

            self.config_dict["Detection"]["vdo_path"] = self.config_dict["Speed_Estimation"]["in_vdo_path"] = str(f)
            with open(self.config_path, 'w') as file:
                json.dump(self.config_dict, file, indent=2, ensure_ascii=False)
            det = TorchDetection(json_path=self.config_path)
            self.det_process = Process(target=det.detect)
            self.det_process.start()
            Log.info("Upload successfully, starting detection process...")
            return JsonResponse({'success': True})

    def get_progress(self, request):
        if request.method == 'GET':
            stage, curr, all = RequestHandler.read_progress()
            if curr >= all:
                if stage == 1 and self.spd_process is None:
                    spd = Speed(self.config_path)
                    self.spd_process = Process(target=spd.calSpeed)
                    self.spd_process.start()
                    Log.info("Detection & Tracking complete successfully, starting speed estimation process...")

                if stage == 2 and curr == 1:  # All done, return rendered video path
                    cap = cv2.VideoCapture(self.config_dict["Speed_Estimation"]["out_vdo_path"])
                    framerate = cap.get(cv2.CAP_PROP_FPS)
                    trans_vio('violation.json', 'violation-after.json')

                    Log.info("All stages complete!")
                    return JsonResponse({'stage': stage, 'curr': curr, 'all': all,
                                         'vdo': '/vdo/' + self.config_dict["Speed_Estimation"]["out_vdo_path"],
                                         'framerate': framerate})
            return JsonResponse({'stage': stage, 'curr': curr, 'all': all, 'vdo': '', 'framerate': -1})
