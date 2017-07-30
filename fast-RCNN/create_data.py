

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

## Ref : https://github.com/DeepRNN/object_detection



def prepare_train_coco_data(params):
    """ Prepare relevant COCO data for training the model. """
    image_dir, annotation_file, data_dir = params.train_coco_image_dir,
                                           params.train_coco_annotation_file,
                                           params.train_coco_data_dir
    batch_size = params.batch_size
    basic_model = params.basic_model
    num_roi = params.num_roi

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
