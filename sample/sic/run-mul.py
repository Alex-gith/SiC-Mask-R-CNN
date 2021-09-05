#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:00:27 2020

@author: wei
"""


import os
import mytoolsmul  #here
import pandas as pd

# sic.train(dataset='../../dataset/sic/OM',weights='coco')
def train_and_mAP(data_name, run_count, new_weights_folder_name, train=True, rename=True, count_mAP=True):
    
    """train"""
    if train:
        dataset_path = '../../dataset/sic/' + data_name
        weights_name = 'coco'
        os.system('python sicmul.py train --dataset={} --weights={}'.format(dataset_path,weights_name))
        print('here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    """計算mAP"""
    if rename:
        #取得權重放置的資料夾名稱
        logs_name_list = sorted(os.listdir('../../logs')) #先排序使最新的權重位於list的最後一個（因為最新的排序總是在最後）
        #logs下所有資料夾名稱中,如果名子裡面有sic的就把那個資料夾改名
        for name in logs_name_list: 
            if 'sic2021' in name:  #here
                weights_folder_name = name
        #更改資料夾名稱
        logs_path = '../../logs/'
        os.rename(logs_path+weights_folder_name, logs_path+new_weights_folder_name)
    
    if count_mAP:
        #取得權重檔名
        weights_name_list = sorted(os.listdir('../../logs/'+new_weights_folder_name))
        weights_file_name = weights_name_list[-1]
        print('\n\n weights_file_name: {}'.format(weights_file_name))

        #讀取weights檔並建立model
        tools = mytoolsmul.Mytools()  #here
        tools.built_model(new_weights_folder_name, weights_file_name) 
    
        #創建結果存放的data frame
        mAP_df = pd.DataFrame(columns=['data type', 'run count', 'train mAP', 'val mAP'])   #here
#        mAP_df = pd.DataFrame(columns=['data type', 'run count', 'train mAP', 'val mAP', 'test mAP'])
        mAP_df.loc[run_count, 'data type'] = data_name
        mAP_df.loc[run_count, 'run count'] = run_count
        #欲計算資料的資料夾路徑
        account_folder_path = '../../dataset/sic/' + data_name 
        
        for name in ['train','val']:   #here
#        for name in ['train','val','test']:
        # for name in ['test']:
            #AP = tools.mAP_account(account_folder_path, account_folder_name=name)
            mAP = tools.mAP_account(account_folder_path, account_folder_name=name)

            record_name = name + ' mAP' #對應的columns name 
            mAP_df.loc[run_count, record_name] = mAP
        
        csv_name = 'mAP_' + data_name +'-'+str(run_count) #儲存的csv檔名
        save_path = 'mAP//' + data_name +'//' +csv_name
        mAP_df.to_csv(save_path)
        
        return mAP_df



""" 批量計算mAP"""
# for i in range(11,21):
#     name = 'KinD_localnorm-' + str(i)
#     train_and_mAP('KinD_localnorm',run_count=i,\
#                   new_weights_folder_name = name,\
#                       train=False, rename=False)

""" 單次計算mAP"""
#train_and_mAP('b2', run_count= '1-mul-sica-2-3-x-x-0724-10ep-(3)', new_weights_folder_name = 'b2-1-mul-sica-2-3-x-x-0724-10ep-(3)', train=False, rename=False)


""""單次訓練"""
data_type = 'b3'
rep = '1-mul-sica'
tra = '1234'
#tra = 'x'
inf = '123'
#inf = 'x'
#aug = '14'
aug = 'x'
#loss = '12'
loss = 'x'
day = '0813-10ep-(37)'
train_and_mAP(data_name=data_type, run_count=rep+"-"+tra+"-"+inf+"-"+aug+"-"+loss+"-"+day, new_weights_folder_name = data_type+"-"+str(rep)+"-"+str(tra)+"-"+str(inf)+"-"+str(aug)+"-"+str(loss)+"-"+str(day), count_mAP=True, rename=True)


"""合併mAP csv file """
# path = 'mAP//KinD_localnorm'
# file_name = 'sum_mAP_KinD_localnorm'
# tools = mytools.Mytools() 
# rr = tools.merge_csv(folder_path=path ,result_file_name=file_name)

"""批量偵測圖片"""
# result_path = 'media/song/62C8DAAEC8DA8029/wei_RCNN/Mask_RCNN/samples/sic/result/KinD_localnorm_xnorm/test'#結果存放路徑
# #create a list that contain all the testing image path
# test_DIR = '/media/song/62C8DAAEC8DA8029/wei_RCNN/Mask_RCNN/dataset/sic/KinD_localnorm_xnorm/test'

# # path = 'mAP//KinD_localnorm_xnorm'
# # file_name = 'sum_mAP_histogram_localnorm_xnorm'

# tools = mytools.Mytools() 
# w_folde_name = 'KinD_localnorm_xnorm-21'
# w_file_name = 'mask_rcnn_jing_shian_0057.h5'
# tools.built_model(w_folde_name, w_file_name)
# tools.detect_img(img_folder_path=test_DIR, result_dir=result_path)




#整合結果的存放dataframe
#sum_mAP_df = pd.DataFrame(columns=['data type', 'run count', 'train mAP', 'val mAp','test mAP'])
# for type_name in ['localnorm_xnorm']:
#    for rep in range(8,14): 
#        folder_name = type_name +'-'+ str(rep) #ex:original-1
#        one_df = train_and_mAP(data_name=type_name, run_count=rep, new_weights_folder_name=folder_name)
#        sum_mAP_df = pd.concat([sum_mAP_df, one_df])

#    csv_name = 'sum_mAP_' + type_name #儲存的csv檔名
#    save_path = 'mAP//' + type_name +'//' +csv_name
#    sum_mAP_df.to_csv(save_path)
#
#sum_mAP_df.to_csv('0318mAP_org_xnorm_loc')





