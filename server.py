from fastapi import FastAPI, File, UploadFile, Form
from model.outline_model import OUTLINE
from model.bbox_model import BBOX
from typing import Annotated
from PIL import Image
import numpy as np
import yaml
import json
import cv2

app = FastAPI()

@app.get('/')
def home():
    return "Hello World~"

@app.post('/outline')
async def get_outline(file: UploadFile):
    with open('config.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    bbox_model = BBOX(args)
    outline_model = OUTLINE(args)
    
    content = await file.read()
    encoded_img = np.fromstring(content, dtype=np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    bbox = bbox_model.get_bbox(img)
    x1, y1, x2, y2 = bbox
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
    contours = [c[0].tolist() for c in contours]
    
    result = {
        'n_outline': len(contours),
        'outlines': contours
    }
    return result