from ultralytics import YOLO
import cv2

class BBOX:
    def __init__(self, args):
        self.args = args
        self.bbox_model_weight = self.args['bbox_model_weight']
        self.model = YOLO(f'model/{self.bbox_model_weight}')
        self.bbox_classes = self.args['bbox_classes']
    
    def get_bbox(self, img):
        results = self.model(img, classes=self.bbox_classes)
        
        bbox_lst = []
        for result in results:
            xyxy_lst = result.boxes.xyxy.cpu().detach().numpy()
            xyxy_lst = xyxy_lst.astype('int')
            bbox_lst.append(xyxy_lst)
        return bbox_lst