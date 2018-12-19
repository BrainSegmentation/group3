
# coding: utf-8



import numpy as np
from mrcnn import config

class Config1(config.Config):
    """Extension of config class which contains informations for training and inference mode."""
    
    NAME = "Config1"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 10
    NUM_CLASSES = 2 + 1
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    RPN_ANCHOR_SCALES = (80, 120, 160, 200, 240)
    RPN_ANCHOR_RATIOS = [ .5, 1, 2 ]
    RPN_NMS_THRESHOLD = .7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    IMAGE_CHANNEL_COUNT = 2
    
    TRAIN_ROIS_PER_IMAGE = 200
    ROI_POSITIVE_RATIO = .5
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    
    MAX_GT_INSTANCES = 100
    DETECTION_MAX_INSTANCES = 100
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

