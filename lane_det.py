import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm
from imutils.video import FPS

class LaneDetect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)


    def preprocess(self, ori_img):
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path':video_path, 'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.heads.get_lanes(data)

        return data
     

lane_config='/content/drive/MyDrive/SpecialProblems/LaneDetection/lanedet/configs/resa/resa18_tusimple.py'
lane_model_path='/content/drive/MyDrive/SpecialProblems/LaneDetection/lanedet/PreModels/resa_r18_tusimple.pth'
lane_cfg = Config.fromfile(lane_config)
lane_cfg.load_from = lane_model_path
lane_detect = LaneDetect(lane_cfg)
video_path = '/content/drive/MyDrive/SpecialProblems/VehicleProximity/waymoObjectDetection/results/lane/1Vid.mp4'
res_path = '/content/drive/MyDrive/SpecialProblems/VehicleProximity/waymoObjectDetection/results/lane/'
lane_cfg.savedir = res_path
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = FPS().start()
out = cv2.VideoWriter(res_path+'final_output.mp4', cv2.VideoWriter_fourcc('M','P','4','V'), 30, (frame_width,frame_height))
currentframe = 0
prev=[]
while (cap.isOpened() and currentframe<2000):
    ret, img = cap.read()
    if len((np.array(img)).shape) == 0:
        break
    data = lane_detect.preprocess(img)
    data['lanes'] = lane_detect.inference(data)[0]
    lanes = [lane.to_array(lane_cfg) for lane in data['lanes']]
    if not lanes:
      lanes=prev
    for lane in lanes:
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            cv2.circle(data['ori_img'], (x, y), 4, (255, 0, 0), 2)
    prev=lanes
    currentframe += 1
    fps.update()
    out.write(data['ori_img'])
    
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
out.release()

