import os
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork, Actor, HeterophilousGraphDataset, Coauthor
import gdown

import scipy.io
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data
import torch_geometric.transforms as T
import scipy.io
import numpy as np
import random
import torch
import scipy.sparse
import csv
import pandas as pd
import json
from ogb.nodeproppred import NodePropPredDataset
from os import path
from torch_sparse import SparseTensor
# from google_drive_downloader import GoogleDriveDownloader as gdd
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os
from collections import defaultdict

import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, f1_score

from typing import Optional, Callable
import os.path as osp
from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data

from .partition import *


'''
#将节点标签随机划分为 训练集、验证集 和 测试集，并返回对应的索引
def rand_train_test_idx(label, train_prop, vaild_prop, test_prop, ignore_negative = True):
    labeled_nodes = torch.where(label != -1)[0]     #找到标签不是-1的节点

    n = labeled_nodes.shape[0]                      #节点个数
    train_num = int(n * train_prop)                 #训练集个数
    valid_num = int(n * valid_prop)                 #验证集个数
    test_num = int(n * test_prop)                   #测试集个数

    perm = torch.as_tensor(np.random.permutation(n))#1 - n 的随机序列
    train_idx = perm[: train_num]
    valid_idx = perm[train_num : train_num + vaild_num]
    test_idx = perm[train_num + vaild_num : train_num + vaild_num + test_num]

    return {'train': train_idx.numpy(), 'valid': valid_idx.numpy(), 'test': test_idx.numpy()}


#将包含训练集、验证集和测试集节点索引的列表转换为对应的布尔型掩码
def index_to_mask(splits, num_nodes):
    mask_len = len(splits)
    train_mask = torch.zeros(num_nodes, dtype = torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype = torch.bool)
    test_mask = torch.zeros(num_nodes, dtype = torch.bool)

    for i in range(mask_len):
        train_mask[splits[i]['train']] = True
        valid_mask[splits[i]['valid']] = True
        test_mask[splits[i]['test']] = True

    return train_mask, valid_mask, test_mask

'''

#加载数据
def load_dataset(train_valid_test_split, root_dir, dataset_name, per_task_class_num, num_masks = 1):
    assert dataset_name in ('cora', 'citeseer'), 'Invalid dataset'

    if len(train_valid_test_split) != 3:
        print("Invalid splits, will use default split proportioin")
        train_valid_test_split = [0.6, 0.2, 0.2]

    train_prop = train_valid_test_split[0]
    valid_prop = train_valid_test_split[1]
    test_prop = train_valid_test_split[2]
    # num_masks = 1

    if dataset_name in ['cora', 'citeseer']:
        dataset = Planetoid(root = root_dir, name = dataset_name)
        data = dataset[0]                   #数据集只有一张大图，所以只用取出第一个图就行
        # splits_list = [rand_train_test_idx(data.y, train_prop = train_prop, valid_prop = valid_prop, test_prop = test_prop) for _ in range(num_masks)]
        # data.train_mask, data.valid_mask, data.test_mask = index_to_mask(splits_list, data.num_nodes)

    in_dim = data.x.shape[1]
    out_dim = ((data.y.max().item() + 1) // per_task_class_num) * per_task_class_num

    return data, in_dim, out_dim


#将加载的数据按类分割为一个一个任务集（包括训练、验证、测试集）
'''
def class_to_task(data, per_task_class_num, train_prop, valid_prop, test_prop, shuffle_flag = False):
    nodes_num = data.x.shape[0]
    classes_num = data.y.max().item() + 1

    train_mask = torch.zeros(num_nodes, dtype = torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype = torch.bool)
    test_mask = torch.zeros(num_nodes, dtype = torch.bool)

    for class_i in range(classes_num):
        class_i_node_mask = data.y == class_i
        class_i_node_num = class_i_node_mask.sum().item()

        class_i_node_list = torch.where(class_i_node_mask)[0].numpy()
        np.random.shuffle(class_i_node_list)      #打乱节点顺序
        
        #将一个类的节点按比例分为训练、验证、测试集
        train_num = int(class_i_node_num * train_prop)                 #训练集个数
        valid_num = int(class_i_node_num * valid_prop)                 #验证集个数
        test_num = int(class_i_node_num * test_prop)                   #测试集个数

        train_idx = class_i_node_list[: train_num]
        valid_idx = class_i_node_list[train_num : train_num + vaild_num]
        test_idx = class_i_node_list[train_num + vaild_num : train_num + vaild_num + test_num]

        #标记每个节点属于哪一个集合当中
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True

    #计算任务数量，任务数量向上取整，之后判断是否舍去凑不够的最后一组当中的类
    tasks_num = (classes_num + per_task_class_num - 1) // per_task_class_num

   
    label_task = {}      #存储标签与任务id之间的对应
    drop_flag = False    #是否有舍的标志

    classes_ind_list = list(range(classes_num))     #所有类编号列表

    if(shuffle_flag):       #如果要打乱
        classes_ind_list =  random.shuffle(classes_ind_list)

    #给每个类标记属于哪一个任务，label[i] = j -> i类在任务j中
    for task_i in range(tasks_num):
        l = task_i * per_task_class_num
        r = min((task_i + 1) * per_task_class_num, classes_num)  #左闭右开

        if r < (task_i + 1) * per_task_class_num:
            drop_flag = True

        for i in range(l, r):
            label_task[classes_ind_list[i]] = task_i
        
    if drop_flag:
        tasks_num = tasks_num - 1

    tasks = [{"train_mask": torch.zeros_like(train_mask).bool(),
             "valid_mask": torch.zeros_like(valid_mask).bool(),
             "test_mask": torch.zeros_like(test_mask).bool()} for _ in range(tasks_num)]

    #把每个类的每个节点分到对应任务的对应集合中
    for i in range(classes_num):
        #这个类的哪些节点属于训练、验证、测试集中
        class_i_train = train_mask & (data.y == i) 
        class_i_valid = valid_mask & (data.y == i)
        class_i_test = test_mask & (data.y == i)
        task_i = label_task[i]

         #这里说明drop_flag = True，因为task_i 理论上范围为[0, tasks_num - 1],而如果task_i == tasks_num说明在给节点划分任务集编号后发生了 tasks_num - 1
        if task_i == tasks_num:   
            continue    #该类被舍弃了
        
        tasks[task_i]["train_mask"] = tasks[task_i]["train_mask"] | class_i_train
        tasks[task_i]["valid_mask"] = tasks[task_i]["valid_mask"] | class_i_valid
        tasks[task_i]["test_mask"] = tasks[task_i]["test_mask"] | class_i_test

    np.random.shuffle(tasks)

    return tasks
'''

#将加载的数据按类分割为一个一个任务集（包括训练、验证、测试集）将一个子图（x,y,edge_index）先按每个任务中包含的类再分割成不同的子图，一个任务包含一个子图，然后再在子图中分训练、验证、测试集掩码
def class_to_task(data, per_task_class_num, train_prop, valid_prop, test_prop, shuffle_flag = False):
    nodes_num = data.x.shape[0]
    classes_num = data.y.max().item() + 1

    train_mask = torch.zeros(nodes_num, dtype = torch.bool)
    valid_mask = torch.zeros(nodes_num, dtype = torch.bool)
    test_mask = torch.zeros(nodes_num, dtype = torch.bool)
    
    classes_nodes = []                                                  #类i包含的所有节点
    #将节点分为训练、验证、测试集
    for class_i in range(classes_num):
        class_i_node_mask = data.y == class_i                           #子图中属于这一类的所有节点
        class_i_node_num = class_i_node_mask.sum().item()               #所有这一类节点的数量

        class_i_node_list = torch.where(class_i_node_mask)[0].numpy()   #属于这一类的节点列表
        classes_nodes.append(class_i_node_list)                         
        np.random.shuffle(class_i_node_list)      #打乱节点顺序         
        
        #将一个类的节点按比例分为训练、验证、测试集
        train_num = int(class_i_node_num * train_prop)                 #训练集个数
        valid_num = int(class_i_node_num * valid_prop)                 #验证集个数
        test_num = int(class_i_node_num * test_prop)                   #测试集个数

        train_idx = class_i_node_list[: train_num]
        valid_idx = class_i_node_list[train_num : train_num + valid_num]
        test_idx = class_i_node_list[train_num + valid_num : train_num + valid_num + test_num]

        #标记每个节点属于哪一个集合当中
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True

    #计算任务数量，任务数量向上取整，之后判断是否舍去凑不够的最后一组当中的类
    tasks_num = (classes_num + per_task_class_num - 1) // per_task_class_num

    task_classes = [[] for _ in range(tasks_num)]  # 任务i包含了哪些节点的类别
    label_task = {}      #存储标签与任务id之间的对应
    drop_flag = False    #是否有舍的标志

    classes_ind_list = list(range(classes_num))     #所有类编号列表

    if(shuffle_flag):       #如果要打乱
        classes_ind_list =  random.shuffle(classes_ind_list)

    #给每个类标记属于哪一个任务，label[i] = j -> i类在任务j中
    for task_i in range(tasks_num):
        l = task_i * per_task_class_num
        r = min((task_i + 1) * per_task_class_num, classes_num)  #左闭右开

        if r < (task_i + 1) * per_task_class_num:
            drop_flag = True

        for i in range(l, r):
            label_task[classes_ind_list[i]] = task_i
            task_classes[task_i].append(i)

    if drop_flag:
        tasks_num = tasks_num - 1

    tasks = [{"train_mask": torch.zeros_like(train_mask).bool(),
              "valid_mask": torch.zeros_like(valid_mask).bool(),
              "test_mask": torch.zeros_like(test_mask).bool()} for _ in range(tasks_num)]

    #把每个类的每个节点分到对应任务的对应集合中
    for i in range(classes_num):
        #这个类的哪些节点属于训练、验证、测试集中
        class_i_train = train_mask & (data.y == i) 
        class_i_valid = valid_mask & (data.y == i)
        class_i_test = test_mask & (data.y == i)
        task_i = label_task[i]

         #这里说明drop_flag = True，因为task_i 理论上范围为[0, tasks_num - 1],而如果task_i == tasks_num说明在给节点划分任务集编号后发生了 tasks_num - 1
        if task_i == tasks_num:   
            continue    #该类被舍弃了
        
        tasks[task_i]["train_mask"] = tasks[task_i]["train_mask"] | class_i_train
        tasks[task_i]["valid_mask"] = tasks[task_i]["valid_mask"] | class_i_valid
        tasks[task_i]["test_mask"] = tasks[task_i]["test_mask"] | class_i_test

    #把每个任务的所有类的子图保存下到对应任务的对应集合中
    for task_i in range(tasks_num):
        nodes_list = []
        for class_idx in task_classes[task_i]:
            nodes_list.extend(classes_nodes[class_idx])
        sub_graph = get_subgraph_by_node(data, nodes_list)

        tasks[task_i]["local_data"] = sub_graph

    ##打乱任务，不打乱就是所有客户端第i个任务使用相同的类进行训练
    # np.random.shuffle(tasks)  

    return tasks
    
#获得每个客户端的数据（分割子图后将子图数据转为任务集）
def get_client_task(data, clients_num, per_task_class_num, train_prop, valid_prop, test_prop, partition_method = "louvain", shuffle_flag = False):
    #每个客户端的子图
    clients_data = louvain_partitioner(data, clients_num)

    clients_tasks = {client_id: {"data" : None,
                                 "task" : None} for client_id in range(clients_num)}

    known_class_list = []

    for client_i in range(clients_num):
        client_data = clients_data[client_i] #加载数据
        clients_tasks[client_i]["data"] = client_data

        #分割任务
        client_tasks = class_to_task(client_data, per_task_class_num, train_prop, valid_prop, test_prop, shuffle_flag)
        clients_tasks[client_i]["task"] = client_tasks

        for task_i in client_tasks:
            client_i_task_i_mask = task_i["train_mask"] | task_i["valid_mask"] | task_i["test_mask"]
            client_i_task_i_known_classes = torch.unique(client_data.y[client_i_task_i_mask])
            known_class_list.append(client_i_task_i_known_classes)

        print(f"client {client_i} has {len(clients_tasks[client_i]['task'])} tasks.")

    known_class = torch.unique(torch.hstack(known_class_list))
    classes_used_num = known_class.shape[0]

    in_dim = data.x.shape[1]
    out_dim = classes_used_num

    if classes_used_num != data.y.max().item() + 1:
        print(f"DROPS {data.y.max().item() + 1 - classes_used_num} CLASS(ES).")

    return clients_tasks, in_dim, out_dim