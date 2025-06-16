from ultralytics import YOLO
import numpy as np
import torch
import yaml
import cv2
import os

class PoseEstimator:
    def __init__(self, args: dict):
        self.args = args
        self.checkpoint = self.args['pose_estimator_checkpoint']
        self.model = YOLO(f'model/{self.checkpoint}')
        self.model.eval()
    
    @torch.no_grad()
    def get_keypoints(self, img: np.ndarray):
        results = self.model(img)
        if len(results) != 1: raise
        result = results[0]
        
        max_area = -np.inf
        for box, keypoint in zip(result.boxes, result.keypoints):
            _, _, w, h = box.xywhn[0]
            area = w * h
            
            if area > max_area:
                max_area = area
                max_keypoint = keypoint.xy[0].cpu().detach().numpy()
                max_xyxy = box.xyxy[0].cpu().detach().numpy()
        return max_xyxy, max_keypoint