from model.outline_model import BaseOUTLINE, OUTLINE
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
    if args['sm_version'] == 1:
        outline_model = BaseOUTLINE(args)
    else:
        outline_model = OUTLINE(args)
    
    folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    if folder_name not in 'result':
        os.makedirs(folder_name)
    
    ext = ['*.jpg', '*.png']
    dataset = args['dataset']
    
    img_files = []
    for t in ext:
        img_files.extend(glob(f'data/{dataset}/{t}'))
    
    ### test
    # img_files = glob(f'data/dataset2/KakaoTalk_20250525_215241922_13.jpg')
    
    for i, img_file in enumerate(img_files):
        ### preprocessing
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rectangle_img = img.copy()
        result_img = img.copy()
        
        ### get bbox & outline
        bbox = bbox_model.get_bbox(img)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(rectangle_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        mask = outline_model.get_outline(img, bbox)
        if len(mask) != 1:
            print(len(mask))
            raise
        
        mask = np.array(mask, dtype=np.uint8)
        mask[mask == 1] = 255
                
        kernel = np.zeros((10, 10), dtype=np.uint8)
        mask_ = cv2.morphologyEx(mask[0], cv2.MORPH_CLOSE, kernel)
        
        _, thresh = cv2.threshold(mask_, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        ### save result images
        cv2.drawContours(result_img, contours, -1, (255, 0, 0), 20)
        
        plt.figure(figsize=(30, 10))
        
        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(rectangle_img)
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(mask_, 'gray')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(result_img)
        plt.axis('off')
        
        plt.tight_layout()
        
        plt.savefig(f'{folder_name}/{i}.jpg')
        plt.clf()
        plt.close()