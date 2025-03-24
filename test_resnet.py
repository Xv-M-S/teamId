#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-03-21

@author: Xv-M-S
"""
from sklearn.cluster import KMeans
from sacred import Experiment
from sacred.observers import FileStorageObserver

import torchvision.transforms.functional as tf
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch

import cv2 as cv
import numpy as np
import logging
import time
import os




ex = Experiment()
ex.observers.append(FileStorageObserver('experiments'))


_logger = None
def get_logger():
    global _logger
    if _logger is None:
        _logger = configure()
    return _logger
def configure():
    instance_logger = logging.getLogger(__name__)
    instance_logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    instance_logger.addHandler(handler)
    instance_logger.addHandler(console)
    return instance_logger

@ex.config
def config():    
    method = 'resnet' #embedding network

    model_name = 'Resnet18_embeddingall'
    model_version = ''

    test_games = [
    'SNGS-116', 'SNGS-117', 'SNGS-118', 'SNGS-119', 'SNGS-120', 
    'SNGS-121', 'SNGS-122', 'SNGS-123', 'SNGS-124', 'SNGS-125', 
    'SNGS-126', 'SNGS-127', 'SNGS-128', 'SNGS-129', 'SNGS-130', 
    'SNGS-131', 'SNGS-132', 'SNGS-133', 'SNGS-134', 'SNGS-135', 
    'SNGS-136', 'SNGS-137', 'SNGS-138', 'SNGS-139', 'SNGS-140', 
    'SNGS-141', 'SNGS-142', 'SNGS-143', 'SNGS-144', 'SNGS-145', 
    'SNGS-146', 'SNGS-147', 'SNGS-148', 'SNGS-149', 'SNGS-150',
    'SNGS-187', 'SNGS-188', 'SNGS-189', 'SNGS-190', 
    'SNGS-191', 'SNGS-192', 'SNGS-193', 'SNGS-194', 
    'SNGS-195', 'SNGS-196', 'SNGS-197', 'SNGS-198', 
    'SNGS-199', 'SNGS-200'
    ]
    trained_models_dir = 'trained_models/'
    
    image_w = 64
    image_h = 128

    image_size = (64,128)

################### resnet-encoder ###############################
class ResNet18Encoder(nn.Module):
    def __init__(self, original_model):
        super(ResNet18Encoder, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])  # 去掉最后的池化层和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 添加全局平均池化层

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平为二维数组
        return x
    
    def encode(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平为二维数组
        return x

def load_model_embed(model_path):  
    #load feature extraction model
    resnet18 = models.resnet18(pretrained=True)
    model = ResNet18Encoder(resnet18)   
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.to(torch.device('cuda'))
    else:
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()

    return model 

def read_and_process(image_path, image_size = (64, 128), top_portion=False):
    # print(image_path)
    image = cv.imread(image_path) 

    temp = image
    if top_portion:
        temp=image[:64,:,:]

    non_black_pixels_mask = np.any(temp != [0, 0, 0], axis=-1)
    extracted = temp[non_black_pixels_mask].tolist()
    if (image is None) or len(extracted)==0:
        return None
    
    image = cv.resize(image, image_size)
        
    # normalize image to change brightness
    image = cv.normalize(image,None,0.0,255.0,norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)
    image = image.astype(np.uint8)
    
    return image



def read_test_image_data(game_path, images_sub_dir):
    image_size = (64, 128)

    gt = open(os.path.join(game_path, 'gt.txt'), 'r')
    lines = gt.readlines()

    images = []
    names = []
    gt_clusters= [[] for x in range(2)]
       
    for line in lines:
        g_tmp = line.split(',')
        # 我们的标签是 0 left 1 right 2 referee
        gt_k = int(g_tmp[1])
        if gt_k >= 2:
            continue
        name_ = str(g_tmp[0]) 
        name  = os.path.join(game_path, images_sub_dir, name_)
        image = read_and_process(name, image_size)        
        if (image is None):
            continue

        gt_clusters[gt_k].append(name)
        names.append(name)
        images.append(image) 
    
    return images, gt_clusters, names   

def get_features(images, model):
    X = []
    for image in images:
        image = tf.to_tensor(image) 
        image =  image.unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()
        out = model.encode(image)
        out = out.detach().cpu().numpy()
        X.append(out[0])
    return X

"""
    计算Precision, Recall
"""
def get_resluts(result_clusters, gt_clusters, cluster_n) :
    accuracy_vector = []
    precision_vector = []
    recall_vector = []
    """
    [[0 1]
    [1 0]]
    """
    options=np.matrix('0,1;1,0')
    for o in range(len(options)):
        TP = 0 # TP: True Positive
        FP = 0 # FP: False Positive
        FN = 0 # FN: False Negative
    
        correct_n=0
        total=0 
        opt = options[o,:]

        for i in range(cluster_n): 
            total_in_cluster = len(gt_clusters[i])
            correct_in_cluster=0
                    
            #cluster index varies per run
            r_i=opt[0,i]   
            for img_name in gt_clusters[i]:
                try:
                    indx=result_clusters[r_i].index(img_name)
                    TP = TP + 1
                    correct_in_cluster=correct_in_cluster+1
                except:
                    FN = FN + 1
                    continue
            
            for img_name in result_clusters[r_i]:
                try:
                    indx=gt_clusters[i].index(img_name)
                except:
                    FP = FP + 1
                    continue
            total = total + total_in_cluster
            correct_n = correct_n + correct_in_cluster
        accuracy_vector.append(correct_n/total)  
        precision_vector.append(TP/(TP+FP))
        recall_vector.append(TP/(TP+FN))    
        logger = get_logger()
        logger.info("TP: " + str(TP) + " FP: " + str(FP) + " FN: " + str(FN) + " total: " + str(total) + " correct_n: " + str(correct_n))
    accuracy = np.max(accuracy_vector)
    precision = np.max(precision_vector)
    recall = np.max(recall_vector)
    print("accuracy: " + str(accuracy))
    print("precision: " + str(precision))  
    print("recall: " + str(recall)) 
        
    return accuracy, precision, recall

def evaluate_clustering_per_game(game, model, images_sub_dir):
    # read all gt
    images, gt_clusters, names = read_test_image_data(game, images_sub_dir)
    # get features
    features = get_features(images, model)
    # cluster
    labels = KMeans(n_clusters=2).fit_predict(features)
    result_clusters= [[] for y in range(2)]
    for m in range(len(labels)):
        result_clusters[labels[m]].append(names[m])  

    # 同时返回accuracy、precsion、recall
    return get_resluts(gt_clusters, result_clusters, 2)

@ex.automain
def main(method, model_name, model_version, test_games, trained_models_dir): 
    logger = get_logger()     
    print("Running with method " + method)
    test_dir = "/home/sxm/data02Space/teamClassfication/teamId/data/test"
    images_sub_dir = 'masked_imgs'

    model_path = trained_models_dir + model_name + model_version + '.pth' 
    model = load_model_embed(model_path)

    #Evaluate method for all test games
    for j, game in enumerate(test_games):
        game_path = os.path.join(test_dir, game)
        mask_path = os.path.join(game_path, images_sub_dir)
        num_images = len(os.listdir(mask_path))
        

        # 记录开始时间
        start_time = time.time()
        logger.info(game + ":(accuracy,precisoin,recall):" + str(evaluate_clustering_per_game(game_path , model, images_sub_dir)))
        # 记录结束时间
        end_time = time.time()
        # 计算运行时间
        elapsed_time = end_time - start_time
        logger.info(f"程序运行时间：{elapsed_time:.4f} 秒")
        logger.info(f"程序fps：{num_images/elapsed_time:.2f} fps")
        