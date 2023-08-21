import itertools
from omegaconf import OmegaConf
import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    DETRDatasetMapper
)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator


dataloader = OmegaConf.create()

train_json_location = None          # coco/annotations/MPSGaze_train_annot_coco_style.json
train_image_location = None         # coco/train2017
valid_json_location = None          # coco/annotations/MPSGaze_val_annot_coco_style.json
valid_image_location = None         # coco/val2017

register_coco_instances(
    "mpsgaze_train",
    {},
    train_json_location,
    train_image_location,
)
register_coco_instances(
    "mpsgaze_val",
    {},
    valid_json_location,
    valid_image_location,
)

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="mpsgaze_train"),
    mapper=L(DETRDatasetMapper)(
        augmentation=[],
        augmentation_with_crop=None,
        is_train=True,
        mask_on=False,
        img_format="RGB",
    )
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="mpsgaze_val", filter_empty=False),
    mapper=L(DETRDatasetMapper)(
        augmentation=[],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False
    )
)
