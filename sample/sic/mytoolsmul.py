#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:53:35 2020

@author: wei
"""

import os
import sys
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import time
import pandas as pd


class Mytools():
    def built_model(self, weights_folder_name, weights_file_name):
        # Root directory of the project
        self.ROOT_DIR = os.path.abspath("../../") #專案的資料夾
        
        # Import Mask RCNN
        sys.path.append(self.ROOT_DIR)  # To find local version of the library
        from my_mrcnn import utils
        import my_mrcnn.model as modellib
        from my_mrcnn import visualize 
        import sicmul  #here
        
        self.sicmul = sicmul  #here
        # detect_img會用到
        self.visualize = visualize 
        self.utils = utils
        # evaluate_model會用到
        self.modellib = modellib
        
        
        # Import jing_shian config
        # 匯入coco資料集，即下載5Kminival和35K validataon-minus-minival子集 （放入coco資料夾中）
        sys.path.append(os.path.join(self.ROOT_DIR, "samples/sic/"))  # To find local version
        
        
        #%matplotlib inline (for jupyter notebook)
        
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        
        # Local path to trained weights file
        #權重檔案路徑要在這裡改
        weights_path = weights_folder_name +'/'+ weights_file_name
        sic_MODEL_PATH = os.path.join(MODEL_DIR,weights_path)
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(sic_MODEL_PATH):
            print("我說權重呢?")
            #utils.download_trained_weights(COCO_MODEL_PATH)
    
        """Configurations
        We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the CocoConfig class in coco.py.
        
        For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the CocoConfig class and override the attributes you need to change."""
        class InferenceConfig(self.sicmul.sicConfig):  #here
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            

        
        self.config = InferenceConfig()
        self.config.display()
        
        
        """Create Model and Load Trained Weights"""
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)
        
        # Load weights trained on MS-COCO
        self.model.load_weights(sic_MODEL_PATH, by_name=True)
        
        """Class Names"""
        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'bpd','tsd','ted'] #here
#        self.class_names = ['BG', 'tsd','bpd','ted'] #here
#        self.class_names = ['BG','tsd']
        
        def get_ax(self, rows=1, cols=1, size=16):
            """Return a Matplotlib Axes array to be used in
            all visualizations in the notebook. Provide a
            central point to control graph sizes.
            
            Adjust the size attribute to control how big to render images
            """
            _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
            return ax



        

    def detect_img(self, img_folder_path, result_dir):
        t_start = time.time()
        
        """Run Object Detection"""
            
        # Directory of images to run detection on
        #IMAGE_DIR = os.path.join(ROOT_DIR, "dataset/image_paths/train")#測試圖片路徑
        # result_dir = os.path.join(ROOT_DIR,'samples/jing_shian/result/0311_val')#結果存放路徑
        
        #create a list that contain all the testing image path
        # test_DIR = os.path.join(ROOT_DIR,"dataset/jing_shian_0308/val")
        image_paths=[]
        for filename in os.listdir(img_folder_path):
            if os.path.splitext(filename)[1].lower() in ['.png','.jpg','.jpeg']:#make sure it is image file
                image_paths.append(os.path.join(img_folder_path,filename))
        
        
        sum_=0
        index=0
        print("total image :{total} ".format(total=len(image_paths)))
        for i ,image_path in enumerate(image_paths):    
            cnt = 0
            index += 1
            img = skimage.io.imread(image_path)
            img_arr = np.array(img)
            results = self.model.detect([img_arr],verbose=0)
            r = results[0]
            ax = self.get_ax(1)
            cnt = (r['class_ids'] == 1).sum() 
            self.visualize.display_instances(img,r['rois'],r['masks'],r['class_ids']
            ,self.class_names,r['scores'],ax=ax,figsize=(5,5),title="Predictions")
            #儲存圖片
            os.chdir(result_dir)
            plt.savefig(image_path.split("/")[-1])
            print("#Image{index}>>{ids} contains {count} sic".\
                  format(index=index,count=cnt,ids=image_path.split("/")[-1]))
            print("-------------------------------")
            sum_+=cnt
        print("total image :{total} ".format(total=len(image_paths)))
        print("Total sic:{sum_}".format(sum_=sum_))
        t_end = time.time()
        print("Spend {time} seconds".format(time=t_end-t_start))


    # 計算給定數據集中模型的 mAP
    def mAP_account(self, account_folder_path, account_folder_name):
        
        account_dataset = self.sicmul.sicDataset()  #here
        account_dataset.load_sic(account_folder_path, account_folder_name)
        account_dataset.prepare()
        
#        print('\naccouting mAP of {} set'.format(str(account_dataset.subset_name)))
        APs = list()
        for i ,image_id in enumerate(account_dataset.image_ids):
            # 加載指定 image id 的圖像、邊框和掩膜
            
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                self.modellib.load_image_gt(account_dataset, self.config,
                                   image_id, use_mini_mask=False)
            molded_images = np.expand_dims(self.modellib.mold_image(image, self.config), 0)
            # Run object detection
            results = self.model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps =\
                self.utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            # 保存
            APs.append(AP)
            #進度條的部份
            rate = int(i/(len(account_dataset.image_ids)-1)*100)
            sys.stdout.write('\r')
            sys.stdout.write("[%s%s] %d%%" % ('='*rate,'-'*(100-rate),rate))
            sys.stdout.flush()
            
        # 計算所有圖片的平均 AP
        mAP = np.mean(APs)
        return mAP ,APs, precisions, recalls, overlaps


    def merge_csv(self, folder_path, result_file_name):
        #創建結果存放的data frame
        sum_mAP_df = pd.DataFrame(columns=['data type', 'run count', 'train mAP', 'val mAP', 'test mAP'])
        #取得folder_path下所有的檔名
        file_list = os.listdir(folder_path)
        print('start merging {} csv'.format(len(file_list)))
        for file_name in file_list:
            one_file = pd.read_csv(folder_path+'/'+file_name)
            sum_mAP_df = pd.concat([sum_mAP_df, one_file]) 
            
        #解決sum_mAP_df中index0全部為的問題
        sum_mAP_df.reset_index(drop=True, inplace=True)
        
        #計算average(不在同一個for放入df,避免計算avg時有nan)
        avg_list = []
        for col in ['train mAP', 'val mAP', 'test mAP']:
            avg = np.average(sum_mAP_df.loc[1::, col])
            avg_list.append(avg)
            
        #放入df    
        for i,col in enumerate(['train mAP', 'val mAP', 'test mAP']):
            sum_mAP_df.loc['average', col] = avg_list[i]
        
        save_path = folder_path + '//' + result_file_name
        sum_mAP_df.to_csv(save_path)









