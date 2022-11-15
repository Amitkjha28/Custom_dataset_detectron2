from detectron2.detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.detectron2.data.datasets import register_coco_instances
from detectron2.detectron2.engine import DefaultTrainer
import os
import pickle

from utils import*

config_file_path = "C:/ Users/amitcc/PycharmProjects/Custom_dataset_detectron2/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "C:/ Users/amitcc/PycharmProjects/Custom_dataset_detectron2/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

output_dir = "./output/Object_Detection"
num_classes = 1
device = "cpu"

train_dataset_name = "Tree_train"
train_images_path = "train"
train_json_annot_path = "train.json"

test_dataset_name = "Tree_test"
test_images_path = "test"
test_json_annot_path = "test.json"

# register_dataset
register_coco_instances(name=train_dataset_name,metadata={},json_file=train_json_annot_path,image_root=train_images_path)

register_coco_instances(name=test_dataset_name,metadata={},json_file=test_json_annot_path,image_root=test_images_path)
plot_samples(dataset_name=train_dataset_name,n=2)
