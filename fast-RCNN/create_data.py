from coco import COCO
import os
import numpy as np

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

## Ref : https://github.com/DeepRNN/object_detection


train_coco_image_dir = "/home/sree/code/rcnn-tensorflow/fast-RCNN/data/train2014"
train_coco_annotation_file = "/home/sree/code/rcnn-tensorflow/fast-RCNN/annotations/instances_train2014.json"
train_coco_data_dir = "/home/sree/code/rcnn-tensorflow/fast-RCNN/data"




coco_num_class = 81

coco_class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush', 80: 'background'}

coco_class_colors = [[225, 239, 163], [202, 196, 172], [252, 182, 134], [170, 148, 215], [216, 243, 246], [229, 150,  89], [223, 226, 140], [154, 159, 166], [ 89, 146, 182], [199, 250, 161], [113, 233, 109], [135, 232,  89], [138, 216, 217], [ 87, 205, 191], [201, 106, 135], [158, 198, 159], [169, 147, 118], [187,  85, 107], [156,  97,  93], [176,  93, 108], [214, 190, 200], [212, 173, 198], [195, 188, 100], [162, 189, 192], [250, 122, 240], [122, 249, 106], [ 96, 110,  87], [230, 177, 203], [250, 201,  81], [195, 220, 198], [ 82, 143,  88], [ 96,  95, 105], [243, 153, 221], [153, 127,  81], [143, 211, 223], [188,  96, 250], [236, 233, 151], [185, 131, 198], [202, 232, 165], [188, 101, 213], [175, 184, 238], [223, 218, 245], [136, 210, 213], [156, 248,  85], [ 93, 221, 116], [200, 253,  91], [130, 210, 103], [210, 102, 212], [180, 178, 197], [160, 115, 138], [186, 229, 120], [184, 107,  86], [117, 229, 229], [186,  96, 139], [183, 215, 253], [106,  86, 154], [159, 184, 236], [217, 217, 194], [171, 108, 147], [ 94, 118, 231], [144, 242, 113], [183, 149, 230], [ 82,  98, 113], [166, 214, 170], [234, 128, 112], [166, 118, 178], [206, 138, 163], [239, 233, 178], [127, 238, 193], [180, 107, 208], [233, 230, 203], [ 92, 177, 113], [167, 209, 190], [245, 233, 109], [159,  92, 246], [208, 235, 166], [240,  91, 230], [118, 192, 103], [216, 102, 147], [170, 162, 200], [206, 252, 204]]

coco_class_to_category = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90, 80: 100}

coco_category_to_class = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79, 100: 80}



batch_size = 128
basic_model = "vgg16"
num_roi = 100


def prepare_train_coco_data():
    """ Prepare relevant COCO data for training the model. """
    image_dir, annotation_file, data_dir = train_coco_image_dir, train_coco_annotation_file, train_coco_data_dir

    coco = COCO(annotation_file)

    img_ids = list(coco.imgToAnns.keys())
    img_files = []
    img_heights = []
    img_widths = []
    anchor_files = []
    gt_classes = []
    gt_bboxes = []

    for img_id in img_ids:

        img_files.append(os.path.join(image_dir, coco.imgs[img_id]['file_name'])) 
        img_heights.append(coco.imgs[img_id]['height']) 
        img_widths.append(coco.imgs[img_id]['width']) 
        
        # import ipdb
        # ipdb.set_trace()

        anchor_files.append(os.path.join(data_dir, os.path.splitext(coco.imgs[img_id]['file_name'])[0]+'_'+basic_model+'_anchor.npz')) 

        classes = [] 
        bboxes = [] 
        for ann in coco.imgToAnns[img_id]: 
            classes.append(coco_category_to_class[ann['category_id']]) 
            bboxes.append([ann['bbox'][1], ann['bbox'][0], ann['bbox'][3]+1, ann['bbox'][2]+1]) 

        gt_classes.append(classes)  
        gt_bboxes.append(bboxes) 
 
    print("Building the training dataset...")
    dataset = DataSet(img_ids, img_files, img_heights, img_widths, batch_size, anchor_files, gt_classes, gt_bboxes, True, True)
    print("Dataset built.")
    return coco, dataset







class DataSet():
    def __init__(self, img_ids, img_files, img_heights, img_widths, batch_size=1, anchor_files=None, gt_classes=None, gt_bboxes=None, is_train=False, shuffle=False):
        self.img_ids = np.array(img_ids)
        self.img_files = np.array(img_files)
        self.img_heights = np.array(img_heights)
        self.img_widths = np.array(img_widths)
        self.anchor_files = np.array(anchor_files)
        self.batch_size = batch_size
        self.gt_classes = gt_classes
        self.gt_bboxes = gt_bboxes
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.current_index = 0
        self.count = len(self.img_files)
        self.indices = list(range(self.count))
        self.num_batches = int(self.count/self.batch_size)
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
