from model.outline_model import OUTLINE
from matplotlib import pyplot as plt
from model.bbox_model import BBOX
from datetime import datetime
from glob import glob
import numpy as np
import yaml
import cv2
import os

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    bbox_model = BBOX(args)
    outline_model = OUTLINE(args)
    
    folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    if folder_name not in 'result':
        os.makedirs(folder_name)
    
    img_files = glob('data/*.jpg')
    for i, img_file in enumerate(img_files):
        ### preprocessing
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rectangle_img = img.copy()
        result_img = img.copy()
        
        
        ### get bbox & outline
        bbox_lst = bbox_model.get_bbox(img)
        if len(bbox_lst) > 1: raise
        x1, y1, x2, y2 = bbox_lst[0][0]
        cv2.rectangle(rectangle_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        mask_lst = outline_model.get_outline(img, bbox_lst)
        
        ### postprocessing
        for j, mask in enumerate(mask_lst):
            mask = np.array(mask[0], dtype=np.uint8)
            mask[mask == 1] = 255
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            ### save result images
            cv2.drawContours(result_img, contours, -1, (255, 0, 0), 20)
            
            plt.figure(figsize=(16, 9))
            
            plt.subplot(1, 4, 1)
            plt.imshow(img)
            plt.axis('off')
            
            plt.subplot(1, 4, 2)
            plt.imshow(rectangle_img)
            plt.axis('off')
            
            plt.subplot(1, 4, 3)
            plt.imshow(mask, 'gray')
            plt.axis('off')
            
            plt.subplot(1, 4, 4)
            plt.imshow(result_img)
            plt.axis('off')
            
            plt.savefig(f'{folder_name}/{i}_{j}.jpg')