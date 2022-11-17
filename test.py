from detectron2.detectron2.engine import DefaultPredictor

import os
import pickle
from utils import*
cfg_save_path = "OD_cfg.pickle"

with open(cfg_save_path,'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

image_path = "test/"
video_path = ""

on_image(image_path, predictor)
on_video(video_path,predictor)



