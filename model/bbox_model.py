from ultralytics import YOLO
import numpy as np

class BBOX:
    def __init__(self, args):
        self.args = args
        self.bbox_model_weight = self.args['bbox_model_weight']
        self.model = YOLO(f'model/{self.bbox_model_weight}')
        self.bbox_classes = self.args['bbox_classes']
    
    def get_bbox(self, img):
        res = self.model(img, classes=self.bbox_classes)
        if len(res) != 1: raise
        
        results = res[0]
        max_bbox = None
        max_area = -np.inf
        for i, result in enumerate(results):
            _, _, w, h = result.boxes.xywhn.cpu().detach().numpy()[0]
            area = w * h
            
            if area > max_area:
                max_area = area
                max_bbox = result.boxes.xyxy.cpu().detach().numpy()[0]
            
            ### if you want to image with bounding boxes
            # result.save(f'data/{i}.png')
        
        bbox = max_bbox.astype('int')
        return bbox