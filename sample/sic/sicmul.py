"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

#import os
#import sys
#import json
#import datetime
#import numpy as np
#import skimage.draw
#
#import matplotlib # bai
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#
## Root directory of the project
#ROOT_DIR = os.path.abspath("../../")
#
## Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
#from mrcnn.config import Config
#from mrcnn import model as modellib, utils
#
#from mrcnn import visualize # bai
#from mrcnn.visualize import display_images # bai
#
#
## Path to trained weights file
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#
## Directory to save logs and model checkpoints, if not provided
## through the command line argument --logs
#DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

import matplotlib#GAI
import matplotlib.pyplot as plt#GAI
import matplotlib.patches as patches#GAI

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from my_mrcnn.config import Config#GAI
from my_mrcnn import model as modellib, utils#GAI

from my_mrcnn import visualize #GAI
from my_mrcnn.visualize import display_images#GAI

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
############################################################
#  Configurations
############################################################



class sicConfig(Config):#GAI
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "sic"#GAI

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + sic

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 50


#=========================================================
#train hyper-parameter

    #1
    BACKBONE = "resnet50"

    #2
    #Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    #3
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 500

    
    #4
    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.66
##    ROI_POSITIVE_RATIO = 0.8
##    ROI_POSITIVE_RATIO = 1
    
#inference hyper-parameter

    #1
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 500
#    
#
#
    #2
    #Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.5
##    DETECTION_NMS_THRESHOLD = 0.7
##    DETECTION_NMS_THRESHOLD = 0.9
#
    #3
    #Minimum probability value to accept a detected instance
    #ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0
##    DETECTION_MIN_CONFIDENCE = 0.3
##    DETECTION_MIN_CONFIDENCE = 0.5
    

#====================================================================
#    LOSS_WEIGHTS = {
#        "rpn_class_loss": 1.,
#        "rpn_bbox_loss": 1.,
#        "mrcnn_class_loss": 1.,
#        "mrcnn_bbox_loss": 1.,
#        "mrcnn_mask_loss": 1.
#    }


#    #1
#    BACKBONE = "resnet50"
#
#    #2
#    #Length of square anchor side in pixels
#    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
#
#
#    #3
#    # Maximum number of ground truth instances to use in one image
#    MAX_GT_INSTANCES = 500
#
#    
#    #4
#    # Percent of positive ROIs used to train classifier/mask heads
#    ROI_POSITIVE_RATIO = 0.66
#    
#    
#
#
#    #1
#    # Max number of final detections
#    DETECTION_MAX_INSTANCES = 500
#    
#
#
#    #2
#    #Non-maximum suppression threshold for detection
#    DETECTION_NMS_THRESHOLD = 0.2
#
#    #3
#    #Minimum probability value to accept a detected instance
#    #ROIs below this threshold are skipped
#    DETECTION_MIN_CONFIDENCE = 0.5
    

############################################################
#  Dataset
############################################################

class sicDataset(utils.Dataset): # bai

    def load_sic(self, dataset_dir, subset): # bai
        """Load a subset of the sic dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
#        self.add_class("sic", 1, "BPD") # bai
#        self.add_class("sic", 2, "TSD") # bai
#        self.add_class("sic", 3, "TED") # bai
        self.add_class("sic", 1, "bpd") # bai
        self.add_class("sic", 2, "tsd") # bai
        self.add_class("sic", 3, "ted") # bai


        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                objects = [s['region_attributes'] for s in a['regions'].values()]  # bai
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                objects = [s['region_attributes'] for s in a['regions']]  # bai

            class_ids = [int(n['sic']) for n in objects] # bai
            #class_ids = [int(n['defect']) for n in objects] # bai

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            # print("multi_numbers=", multi_numbers)
            # num_ids = [n for n in multi_numbers['number'].values()]
            # for n in multi_numbers:

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "sic",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)   # bai

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "sic":
            return super(self.__class__, self).load_mask(image_id)
        class_ids = image_info['class_ids']  # bai
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # class_ids=np.array([self.class_names.index(shapes[0])])
       # print("info['class_ids']=", info['class_ids'])
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids  # [mask.shape[-1]] #np.ones([mask.shape[-1]], dtype=np.int32)#class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sic": # bai
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = sicDataset() # bai
    dataset_train.load_sic(args.dataset, "train") # bai
    dataset_train.prepare()

    # Validation dataset
    dataset_val = sicDataset() # bai
    dataset_val.load_sic(args.dataset, "val") # bai
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.

#==========
#     augmentation = iaa.SomeOf((0, None), [
#         iaa.Fliplr(1.0),
#         iaa.Flipud(1.0),
#         iaa.OneOf([iaa.Affine(rotate=90),
#                     iaa.Affine(rotate=180),
#                     iaa.Affine(rotate=270)]), #1
# #        iaa.Multiply((0.9, 1.1),per_channel=1.0), #2 (0.8,1.2)
# #        iaa.GammaContrast((0.8, 1.25)), #3 (0.75,1.3)
#         iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)) #4
#     ], random_order=True)
#==========

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10, # 30,  bai
#                augmentation=augmentation, #here
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def color_show(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


def get_ax(rows=1, cols=1, size=16): # bai
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def detect_and_show(model, image_path=None, video_path=None): #bai
    import visualize_cv2  # bai
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Validation dataset
        dataset = sicDataset()  # bai
        dataset.load_sic(args.dataset, "val")  # bai

        dataset.prepare()

        print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        results = model.detect([image], verbose=1)
        r = results[0]
        # Save output
        file_name = "detected_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())

        N = r['rois'].shape[0]

#        filter_classs_names = ['BPD', 'TSD', 'TED']
        filter_classs_names = ['bpd', 'tsd', 'ted']
        visualize_cv2.save_image(image, file_name, r['rois'], r['masks'],
                             r['class_ids'], r['scores'], dataset.class_names,
                             filter_classs_names, scores_thresh=0.7, mode=0)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_show(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect sic.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the sic dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == "show": # bai
        assert args.image or args.video,\
               "Provide --image or --video to show detected result"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = sicConfig() # bai
    else:
        class InferenceConfig(sicConfig): # bai
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "show": # bai
        detect_and_show(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
