#yolo classify train data=datasets/garbage model=yolo12n-cls.pt epochs=50 imgsz=224
import cv2
from ultralytics import YOLO

model = YOLO("yolo12n.pt")  # load a pretrained model (recommended for training)

results = model.train(data="E:/PythonProjects/RecycleVision/datasets/garbage", epochs=100, imgsz=640, task='classify')
