import create_data
from image_loader import ImageLoader
import tensorflow as tf
import math

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


class BaseModel(object):
    
    def __init__(self):
        self.num_class = create_data.coco_num_class
        self.class_names = create_data.coco_class_names
        self.class_colors = create_data.coco_class_colors
        self.class_to_category = create_data.coco_class_to_category
        self.category_to_class = create_data.coco_category_to_class
        self.background_id = self.num_class - 1

        
        self.basic_model = "vgg16"
        self.num_roi = "100"
        self.bbox_per_class = False

        self.label = "coco/vggnet16"
        self.img_loader = ImageLoader("ilsvrc_2012_mean.npy")
        self.image_shape = [640, 640, 3]

        self.anchor_scales = [50, 100, 200, 300, 400, 500] 
        self.anchor_ratios = [[1.0/math.sqrt(2), math.sqrt(2)], [1.0, 1.0], [math.sqrt(2), 1.0/math.sqrt(2)]]
        self.num_anchor_type = len(self.anchor_scales) * len(self.anchor_ratios)
        

        ## what is this for ??
        self.anchor_shapes = []
        for s in self.anchor_scales:
            for r in self.anchor_ratios:
                self.anchor_shapes.append([int(s*r[0]), int(s*r[1])])

        self.anchor_stat_file = 'coco_anchor_stats.npz'
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False) 

        # self.build() 
        self.saver = tf.train.Saver(max_to_keep = 100) 
        
