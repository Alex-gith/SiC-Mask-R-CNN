# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:58:40 2020

@author: Grover
"""
#%%
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time



# Root directory of the project
ROOT_DIR = os.path.abspath("F:\\Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/defect/"))  # To find local version
import TSD


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "tsd20200701T1049-config-2/mask_rcnn_tsd_0030.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print("很抱歉，我找不到權重")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "val_set_add_png")

#%%
class InferenceConfig(TSD.TSDConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'TSD']


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

#%%
"""Run Object Detection"""
t_start = time.time()
# Directory of images to run detection on
folder_name = '07131622_TSD_config2'

result_dir = os.path.join('F:\\Mask_RCNN\\result\\' + folder_name)#結果存放路徑

splash_path = "F:\\Mask_RCNN\\result\\" + folder_name + "\\splash"

#create a list that contain all the testing image path
test_DIR = os.path.join("F:\\Mask_RCNN\\test\\test_tsd")#測試圖片路徑
image_paths=[]
for filename in os.listdir(test_DIR):
    if os.path.splitext(filename)[1].lower() in ['.png','.jpg','.jpeg']:#make sure it is image file
        image_paths.append(os.path.join(test_DIR,filename))


os.chdir(result_dir)

sum_=0
index=0
print("total image :{total} ".format(total=len(image_paths)))
for i ,image_path in enumerate(image_paths):    
    cnt = 0
    index += 1
    img = skimage.io.imread(image_path)
    img = skimage.color.grey2rgb(img)
    img_arr = np.array(img)
    results = model.detect([img_arr],verbose=0)
    r = results[0]
    ax = get_ax(1)
    cnt = (r['class_ids'] == 1).sum() 
    visualize.display_instances(img,r['rois'],r['masks'],r['class_ids']
    ,class_names,ax=ax,figsize=(5,5),title="Predictions")
    #儲存圖片
    plt.savefig(image_path.split("\\")[-1])
    print("#Image{index}>>{ids} contains {count} TSD".\
          format(index=index,count=cnt,ids=image_path.split("\\")[-1]))
    print("-------------------------------")
    sum_+=cnt
    image = skimage.io.imread(image_path)
    splash = TSD.color_splash(image, r['masks']) 
    splash_path = result_dir + '\\splash'
    if not os.path.isdir(splash_path):
        os.mkdir(splash_path)
    skimage.io.imsave(result_dir + '\\splash\\' + image_path.split("\\")[-1] + '.png' , splash)
print("total image :{total} ".format(total=len(image_paths)))
print("Total TSD:{sum_}".format(sum_=sum_))
t_end = time.time()
print("Spend {time} seconds".format(time=t_end-t_start))
splash_filenames = os.listdir(splash_path)
base_dir = splash_path + '\\'
reddot_list = []
print('Begin to calculate the area of TSD.')
for i, img in enumerate(splash_filenames):
    img = skimage.io.imread(base_dir + img[0:3] + ".jpg.png")
    a = range(0,512)
    b = range(0,512)
    for j in a:
        for k in b:
            if img[j, k, 0] == 255 and img[j, k, 1] == 0 and img[j, k, 2] == 0:
                w = [j, k]
                reddot_list.append(w)

    # 進度條
    rate = int(i/(len(splash_filenames)-1)*100)
    sys.stdout.write('\r')
    sys.stdout.write("[%s%s] %d%%" % ('='*rate,'-'*(100-rate),rate))
    sys.stdout.flush()

area_total = len(reddot_list)
print('Done.')
print(' Total pixels of TSD are ' + str(area_total) + '.')
t_end = time.time()
print("Spend {time} seconds".format(time=t_end-t_start))


#%%
# 計算給定數據集中模型的 mAP
def evaluate_model(dataset, model, cfg):
    print('accouting mAP of {} set'.format(str(dataset.subset_name)))
    APs = list()
    for i ,image_id in enumerate(dataset.image_ids):
        # 加載指定 image id 的圖像、邊框和掩膜
        
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, cfg,
                               image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        # 保存
        APs.append(AP)
        #進度條的部份
#        rate = (i+1)/len(dataset.image_ids)
#        sys.stdout.write('\r')
#        sys.stdout.write("[%s%s] %d%%" % ('='*i,'-'*(len(dataset.image_ids)-i-1),rate*100))
#        sys.stdout.flush()
        
        rate = int(i/(len(dataset.image_ids)-1)*100)
        sys.stdout.write('\r')
        sys.stdout.write("[%s%s] %d%%" % ('='*rate,'-'*(100-rate),rate))
        sys.stdout.flush()
        
    # 計算所有圖片的平均 AP
    mAP = np.mean(APs)
    return mAP


#
#train_set = pothole.potholeDataset()
#train_set.load_pothole('/home/song/Mask_RCNN/dataset/pothole_0308','train')#待計算數據集之路徑,數據集名稱
#train_set.prepare()
#train_mAP = evaluate_model(train_set, model, config)
#print("\nTrain mAP: %.3f" % train_mAP)

test_set = TSD.TSDDataset()
test_set.load_TSD('F:\\Mask_RCNN\\dataset\\defect','train')#待計算數據集之路徑,數據集名稱
test_set.prepare()
test_mAP = evaluate_model(test_set, model, config)
print("\nTest mAP: %.3f" % test_mAP)





#%%
"""Run Object Detection 備份"""
t_start = time.time()
# Directory of images to run detection on
folder_name = '06191026'

result_dir = os.path.join('F:\\Mask_RCNN\\result\\' + folder_name)#結果存放路徑

splash_path = "F:\\Mask_RCNN\\result\\" + folder_name + "\\splash"

#create a list that contain all the testing image path
test_DIR = os.path.join("F:\\Mask_RCNN\\test\\test_100")#測試圖片路徑
image_paths=[]
for filename in os.listdir(test_DIR):
    if os.path.splitext(filename)[1].lower() in ['.png','.jpg','.jpeg']:#make sure it is image file
        image_paths.append(os.path.join(test_DIR,filename))


os.chdir(result_dir)

sum_=0
sum2_ =0
index=0
print("total image :{total} ".format(total=len(image_paths)))
for i ,image_path in enumerate(image_paths):    
    cnt = 0
    index += 1
    img = skimage.io.imread(image_path)
    img = skimage.color.grey2rgb(img)
    img_arr = np.array(img)
    results = model.detect([img_arr],verbose=0)
    r = results[0]
    ax = get_ax(1)
    cnt = (r['class_ids'] == 1).sum()
    cnt2 = (r['class_ids'] == 2).sum() 
    visualize.display_instances(img,r['rois'],r['masks'],r['class_ids']
    ,class_names,ax=ax,figsize=(5,5),title="Predictions")
    #儲存圖片
    plt.savefig(image_path.split("\\")[-1] + '.png')
    print("#Image{index}>>{ids} contains {count} TSD & {count2} TSD".\
          format(index=index,count=cnt,count2=cnt2,ids=image_path.split("\\")[-1]))
    print("-------------------------------")
    sum_+=cnt
    sum2_+=cnt2
    image = skimage.io.imread(image_path)
    splash = TSD.color_splash(image, r['masks']) 
    splash_path = result_dir + '\\splash'
    if not os.path.isdir(splash_path):
        os.mkdir(splash_path)
    skimage.io.imsave(result_dir + '\\splash\\' + image_path.split("\\")[-1] + '.png' , splash)
print("total image :{total} ".format(total=len(image_paths)))
print("Total TSD:{sum_}, TSD:{sum2_}".format(sum_=sum_, sum2_=sum2_))
t_end = time.time()
print("Spend {time} seconds".format(time=t_end-t_start))

#%%
'''
以下為測試用
'''
TSD_DIR = os.path.join(ROOT_DIR, "dataset/TSD")
# Load validation dataset
dataset = TSD.TSDDataset()
dataset.load_TSD(TSD_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

#%%
result_dir = os.path.join('F:\\Mask_RCNN\\result\\06261200')#結果存放路徑
os.chdir(result_dir)
#image_id = random.choice(dataset.image_ids)
image_id = 1
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, ax=ax,
                            title="Predictions")
plt.savefig(dataset.image_reference(image_id).split("\\")[-1])
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)


#%%

splash = TSD.color_splash(image, r['masks']) 
display_images([splash], cols=1)
skimage.io.imsave(result_dir + '\\output.png', splash)

#%%

if class_id == 1:
            color = (0.0, 1.0, 0.0)
            if show_mask:
                masked_image = apply_mask(masked_image, mask, color)
        # TSD
        elif class_id == 2:
            color = (1.0, 0.0, 0.0)
            if show_mask:
                masked_image = apply_mask(masked_image, mask, color)

#%%

TSD.detect_and_color_splash(COCO_MODEL_PATH, test_DIR) 

#%%
img_red = np.zeros([512, 512, 3], np.uint8)
img_red[:, :, 0] = np.zeros([512, 512])+ 255
display_images([img_red], cols=1)

