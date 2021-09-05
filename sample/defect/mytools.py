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
        import jing_shian
        
        self.jing_shian = jing_shian
        # detect_img會用到
        self.visualize = visualize 
        self.utils = utils
        # evaluate_model會用到
        self.modellib = modellib
        
        
        # Import jing_shian config
        # 匯入coco資料集，即下載5Kminival和35K validataon-minus-minival子集 （放入coco資料夾中）
        sys.path.append(os.path.join(self.ROOT_DIR, "samples/jing_shian/"))  # To find local version
        
        
        #%matplotlib inline (for jupyter notebook)
        
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        
        # Local path to trained weights file
        #權重檔案路徑要在這裡改
        weights_path = weights_folder_name +'/'+ weights_file_name
        jing_shian_MODEL_PATH = os.path.join(MODEL_DIR,weights_path)
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(jing_shian_MODEL_PATH):
            print("我說權重呢?")
            #utils.download_trained_weights(COCO_MODEL_PATH)
    
        """Configurations
        We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the CocoConfig class in coco.py.
        
        For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the CocoConfig class and override the attributes you need to change."""
        class InferenceConfig(self.jing_shian.jing_shianConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        
        self.config = InferenceConfig()
        # self.config.display() #秀出model資訊
        
        
        """Create Model and Load Trained Weights"""
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)
        
        # Load weights trained on MS-COCO
        self.model.load_weights(jing_shian_MODEL_PATH, by_name=True)
        
        """Class Names"""
        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'jing_shian']
        
        
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
            print("#Image{index}:{ids} contains {count} jing_shian".\
                  format(index=index,count=cnt,ids=image_path.split("/")[-1]))
            print("-------------------------------")
            sum_+=cnt
        print("total image :{total} ".format(total=len(image_paths)))
        print("Total jing_shian:{sum_}".format(sum_=sum_))
        t_end = time.time()
        print("Spend {time} seconds".format(time=t_end-t_start))
        
        txt_path = os.path.join(result_dir, 'record')
        with open(txt_path, 'w') as f:
            f.write('total image :{}'.format(len(image_paths)))
            f.write('  Total jing_shian:{}'.format(sum_))


    # 計算給定數據集中模型的 mAP
    def mAP_account(self, account_folder_path, account_folder_name):
        
        account_dataset = self.jing_shian.jing_shianDataset()
        account_dataset.load_jing_shian(account_folder_path, account_folder_name)
        account_dataset.prepare()
        
        print('\naccouting mAP of {} set'.format(str(account_dataset.subset_name)))
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
        return mAP


    def merge_csv(self, backbone, seed, data_type):
        #------基本設置
        #資料夾路徑
        folder_path = os.path.join('mAP', backbone, 'seed_'+str(seed), data_type )
        #sum檔名
        result_file_name = 'sum_mAP_' + backbone + '_s' + str(seed) + '_' + data_type        
        #創建結果存放的data frame
        sum_mAP_df = pd.DataFrame(columns=['data type', 'run count', 'train mAP', 'val mAP', 'test mAP'])
        
        
        #------刪除先前sum檔
        save_path = folder_path + '/' + result_file_name + '.csv'
        save_path = os.path.join(os.getcwd(), save_path)
        if os.path.exists(save_path):
            os.remove(save_path) #先去除先前版本
        
        #------合併csv檔
        #取得folder_path下所有的檔名
        file_list = os.listdir(folder_path)
        print('start merging {} csv'.format(len(file_list)))
        for file_name in file_list:
            one_file = pd.read_csv(folder_path+'/'+file_name)
            sum_mAP_df = pd.concat([sum_mAP_df, one_file]) 
            
            
        #------整理合併csv檔
        #根據run count排序資料,不然會1 10 100 都在前面才換2 20 200
        sum_mAP_df = sum_mAP_df.sort_values(by='run count')   
        #解決sum_mAP_df中index全部為0的問題
        sum_mAP_df.reset_index(drop=True, inplace=True)
        
        
        #------計算統計量
        #計算average(不在同一個for放入df,避免計算avg時有nan)
        avg_list = []
        for col in ['train mAP', 'val mAP', 'test mAP']:
            avg = np.average(sum_mAP_df.loc[:, col])
            avg_list.append(avg)
            
        #計算std(不在同一個for放入df,避免計算std時有nan)
        std_list = []
        for col in ['train mAP', 'val mAP', 'test mAP']:
            std = np.std(sum_mAP_df.loc[:, col])
            std_list.append(std)        
            
        #計算max(不在同一個for放入df,避免計算std時有nan)
        max_list = []
        for col in ['train mAP', 'val mAP', 'test mAP']:
            max_mAP = np.max(sum_mAP_df.loc[:, col])
            max_list.append(max_mAP)           
        
        #放入df    
        for i,col in enumerate(['train mAP', 'val mAP', 'test mAP']):
            sum_mAP_df.loc['mAP avg', col] = avg_list[i]
            sum_mAP_df.loc['mAP std', col] = std_list[i] 
            sum_mAP_df.loc['max mAP', col] = max_list[i] 
            
            
        #------匯出檔案
        sum_mAP_df.to_csv(save_path)
    
    
    def make_block_report(self, backbone, seed):
        
        block_folder_path = os.path.join('mAP', backbone, 'seed_'+str(seed))#資料夾路徑
        type_list = os.listdir(block_folder_path)

        #------block report
        block_report_val = pd.DataFrame()
        block_report_test = pd.DataFrame()        
        
        for type_name in type_list:
            if not "csv" in type_name:#避免把之前的report當成資料夾
                type_folder_path = os.path.join(block_folder_path, type_name)    
            
                # #------確認該資料夾有沒有產生過sum檔
                # have_sum_file = False
                # csv_name_list = os.listdir(type_folder_path)
                
                # for csv_name in csv_name_list:
                #     if "sum" in csv_name:
                #         have_sum_file = True
                # #若沒有 則進行合併
                # if have_sum_file == False:
                #     print('\n{}還沒merge csv喔!!我幫你產生'.format(type_name))
                #     self.merge_csv(backbone, seed, type_name)
                # else:
                #     print('\n{}有sum檔'.format(type_name))
            
            
                #------統一重新產生過sum檔
                print('\n*{}* merge csv'.format(type_name))                
                self.merge_csv(backbone, seed, type_name)

            
                #------讀取sum檔
                sum_csv_path =  os.path.join(type_folder_path, os.listdir(type_folder_path)[-1])    
                df = pd.read_csv(sum_csv_path)    
                df.index = df.loc[:, 'Unnamed: 0']
                # print(df)
                
                #------輸入統計量
                # #val
                # block_report_val.insert(int(last_run_count)  , 'mAP avg', df.loc['mAP avg', 'val mAP'])
                # block_report_val.insert(int(last_run_count)+1, 'mAP std', df.loc['mAP std', 'val mAP'])
                # block_report_val.insert(int(last_run_count)+2, 'max mAP', df.loc['max mAP', 'val mAP'])
                
                block_report_val.loc[type_name, 'mAP avg'] = df.loc['mAP avg', 'val mAP']
                block_report_val.loc[type_name, 'mAP std'] = df.loc['mAP std', 'val mAP']
                block_report_val.loc[type_name, 'max mAP'] = df.loc['max mAP', 'val mAP']

                #test
                block_report_test.loc[type_name, 'mAP avg'] = df.loc['mAP avg', 'test mAP']
                block_report_test.loc[type_name, 'mAP std'] = df.loc['mAP std', 'test mAP']
                block_report_test.loc[type_name, 'max mAP'] = df.loc['max mAP', 'test mAP']
                
                
                for i in df.index:
                    #------輸入每次試驗mAP值
                    #取得run count
                    run_count = str(df.loc[i, 'run count'])
                    if run_count != "nan":#避免取得空白處被讀成nan的值
                        run_count = str(int(float(run_count))) #ex:str(1.0) --> str(1)
                        #val
                        block_report_val.loc[type_name, 'run '+run_count] = df.loc[i, 'val mAP']
                        #test
                        block_report_test.loc[type_name, 'run '+run_count] = df.loc[i, 'test mAP']   
                        
                        # last_run_count = run_count                
                
        #------匯出結果
        #val
        val_result_csv_name = 'val_report_' + backbone + '_s' + str(seed) + '.csv'
        val_result_path = os.path.join(block_folder_path, val_result_csv_name)
        block_report_val.to_csv(val_result_path)
        
        #test
        test_result_csv_name = 'test_report_' + backbone + '_s' + str(seed) + '.csv'
        test_result_path = os.path.join(block_folder_path, test_result_csv_name)
        block_report_test.to_csv(test_result_path)
                
            
            
            
                    


def train_and_mAP(data_name, run_count, seed, backbond='resnet101',
                  train=True, rename=True, count_mAP=True, del_logs=True):
    
    new_weights_folder_name = backbond +'_s' + str(seed) + '_' + data_name + '-' + str(run_count) 
    """train"""
    if train:
        dataset_path = '../../dataset/jing_shian/seed_' + str(seed) +'/'+ data_name
        weights_name = 'coco'
        os.system('python jing_shian.py train --dataset={} --weights={}'.format(dataset_path,weights_name))
        print('here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    """重新命名logs檔"""
    if rename:
        #取得權重放置的資料夾名稱
        logs_name_list = sorted(os.listdir('../../logs')) #先排序使最新的權重位於list的最後一個（因為最新的排序總是在最後）
        #logs下所有資料夾名稱中,如果名子裡面有jing_shian的就把那個資料夾改名
        for name in logs_name_list: 
            if 'jing_shian' in name:
                weights_folder_name = name
        #更改資料夾名稱
        logs_path = '../../logs/'
        os.rename(logs_path+weights_folder_name, logs_path+new_weights_folder_name)
    
    """"
    保留最後一個logs其餘刪除
    False:保留所有log  True:保留最一個  任意整數(ex:10-->10,20..+最後一個)
    """
    if del_logs != False :   
        #讀取weights資料夾中所有檔案
        logs_path = '../../logs/'
        new_weights_folder_path = os.path.join(logs_path, new_weights_folder_name)
        content_list = sorted(os.listdir(new_weights_folder_path))
        
        #將event檔排除考慮
        for name in content_list:
            if 'events' in name:
                 del content_list[content_list.index(name)]
                 break
             
        #經排序後的content_list是由小排到大,因此刪到剩下最後一代epoch的weights為止
        if del_logs == True:
            while len(content_list) > 1:
                del_logs = content_list[0]
                del_logs_path = os.path.join(new_weights_folder_path, del_logs)
                os.remove(del_logs_path)
                del content_list[content_list.index(del_logs)]
        else:
            #取得最後epoch數
            last_logs_name = content_list[-1]
            last_logs_name = last_logs_name.split("_")[-1] #'0100.h5'
            last_logs_name = last_logs_name.split(".")[0] #'0100'
            last_logs_name = int(last_logs_name) #100            
            
            for name in content_list:
                #取得epoch次數
                n = name.split("_")[-1] #'0100.h5'
                n = n.split(".")[0] #'0100'
                n = int(n) #100
                #最後一個epoch一定要留
                if n == last_logs_name :
                    break
                if n % del_logs != 0: #若不是指定留下的logs則刪除
                   del_logs_path = os.path.join(new_weights_folder_path, name)
                   os.remove(del_logs_path) 
        
    """
    * 計算mAP *
    False:不進行計算  
    True:計算mAP 
    str(epoch數):計算指定epoch logs檔 e.g:epoch100-->str(0100)
    """
    if count_mAP != False:
        
        #------取得權重檔名        
        if count_mAP == True:    
            #取最後一個epoch logs檔
            weights_name_list = sorted(os.listdir('../../logs/'+new_weights_folder_name))
            weights_file_name = weights_name_list[-1]
            print('\n\n weights_file_name: {}'.format(weights_file_name))
            #取得epoch num
            epoch_num = weights_file_name.split("_")[-1] #'0100.h5'
            epoch_num = epoch_num.split(".")[0] #'0100'
            
        else:
            #根據指定epoch檔取得logs檔名
            weights_file_name = 'mask_rcnn_jing_shian_' + count_mAP + '.h5'
            print('\n\n weights_file_name: {}'.format(weights_file_name))
            #取得epoch num
            epoch_num = weights_file_name.split("_")[-1] #'0100.h5'
            epoch_num = epoch_num.split(".")[0] #'0100'             

            
            
        #------讀取weights檔並建立model
        tools = Mytools() 
        tools.built_model(new_weights_folder_name, weights_file_name) 
    
        #------創建結果存放的data frame
        mAP_df = pd.DataFrame(columns=['data type', 'run count', 'train mAP', 'val mAP', 'test mAP'])
        mAP_df.loc[run_count, 'data type'] = data_name
        mAP_df.loc[run_count, 'run count'] = run_count
        
        #------欲計算資料的資料夾路徑
        account_folder_path = '../../dataset/jing_shian/seed_' + str(seed) + '/' + data_name 
        
        # for name in ['train','val','test']:
        for name in ['val','test']:
            mAP = tools.mAP_account(account_folder_path, account_folder_name=name)
            
            record_name = name + ' mAP' #對應的columns name 
            mAP_df.loc[run_count, record_name] = mAP
        
        #------儲存結果
        #儲存的csv檔名
        csv_name = 'mAP_'+ backbond +'_s' + str(seed) + '_' +  data_name +'-'+str(run_count) + '(' + epoch_num + ').csv' 
        
        save_folder = 'mAP//'+ backbond +'//seed_' + str(seed) + '//' + data_name 
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        save_path = os.path.join(save_folder, csv_name)        
        
        mAP_df.to_csv(save_path, index=False)
        
        return mAP_df





