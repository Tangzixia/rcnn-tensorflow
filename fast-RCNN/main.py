import create_data
from base_model import BaseModel

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

coco, dataset = create_data.prepare_train_coco_data()

model = BaseModel()
