from model.pose_estimator import PoseEstimator
from fastapi import FastAPI, UploadFile
from model.outline_model import OUTLINE
from model.bbox_model import BBOX
import numpy as np
import yaml
import cv2

app = FastAPI()

KEYPOINTS = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist',
    'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
]

@app.get('/')
def home():
    return "Main Page"

@app.post('/outline')
async def get_outline(file: UploadFile):
    with open('config.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    bbox_model = BBOX(args)
    outline_model = OUTLINE(args)
    pose_estimator = PoseEstimator(args)
    
    content = await file.read()
    encoded_img = np.fromstring(content, dtype=np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    bbox = bbox_model.get_bbox(img)
    mask = outline_model.get_outline(img, bbox)
    _, keypoints = pose_estimator.get_keypoints(img)
    
    if len(mask) != 1:
        print(len(mask))
        raise
    
    mask = np.array(mask, dtype=np.uint8)
    mask[mask == 1] = 255
    
    kernel = np.zeros((10, 10), dtype=np.uint8)
    mask_ = cv2.morphologyEx(mask[0], cv2.MORPH_CLOSE, kernel)
    
    _, thresh = cv2.threshold(mask_, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c.tolist() for c in contours]
    
    result = {
        'n_outline': len(contours),
        'outlines': contours,
        'keypoints': {keypoint:v.tolist() for keypoint, v in zip(KEYPOINTS, keypoints)}
    }
    return result