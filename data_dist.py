import copy
import numpy as np
import math
import random
from itertools import chain
import torch
import torchvision
from torchvision import datasets, transforms

def data_iid(dataset, num_classes, num_nodes):
    """
    Sample I.I.D. client data for the selected dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """ 
    idx_list = {labels:[] for labels in range(num_classes)}
    rng = np.random.default_rng()
    for label in range(num_classes):
        for i in range(len(dataset)):
            if dataset.targets[i] == label:
                idx_list[label].append(i)
        rng.shuffle(idx_list[label])
    
    dict_users = {i:[] for i in range(num_nodes)}
    node_list = list(range(num_nodes))
    random.shuffle(node_list)
    for label in range(num_classes):
        vals = np.array_split(idx_list[label], num_nodes) #Creates num_nodes uniform splits
        for i,node in enumerate(node_list):
            dict_users[node] += list(vals[i])
    for node in range(num_nodes):
        rng.shuffle(dict_users[node])
    del node_list, idx_list, vals 
    return dict_users

def get_dataset_indices(dataset, num_classes):
    """ 
    Return indices for all classes in any Torch dataset.
    Length of idx_list = num_classes as each sub-array contains the list of indices for that class of index.
    """
    idx_list = []
    for class_label in range(num_classes):
        idx = []
        for i in range(len(dataset)):
            if dataset.targets[i] == class_label:
                idx.append(i)
        idx_list.append(idx)
    return idx_list
#     for class_label in range(num_classes):
#         isolate = trg_data.targets == class_label
#         idx = [i for i, val in enumerate(isolate) if val == True]
#         idx_list.append(idx)
#     return idx_list

def data_noniid(dataset, num_labels:int, num_users:int, alpha:float) -> dict:
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data    
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """    
    dist = np.random.dirichlet([alpha] * num_users, num_labels)
    class_ids = [np.argwhere(dataset.targets==y).flatten() for y in range(num_labels)]
    device_dist  =[[] for _ in range(num_users)]
    dict_users = {i:None for i in range(num_users)}
    
    for c, fracs in zip(class_ids, dist):
        for i, ids in  enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            device_dist[i] += [ids]
        
    device_dist = [list(np.concatenate(ids)) for ids in device_dist]
    for node in dict_users.keys():
        np.random.shuffle(device_dist[node])
        dict_users[node] = device_dist[node]
    return dict_users    

def partition (lst:list, n:int) -> list:
    return [ lst[i::n] for i in range(n) ]    
 
def niid_skew_dist(dataset, num_classes:int, num_nodes:int, skew:int) -> dict:
    """
    Generates Skewed Non-IID distributions
    """

    class_ids = [(np.argwhere(dataset.targets==y).flatten()).tolist() for y in range(num_classes)]
    assigned_classes = [random.sample(list(range(num_classes)), skew) for _ in range(num_nodes)]
    newlist = list(chain.from_iterable(assigned_classes))
    class_counts = {i:newlist.count(i) for i in range(num_classes)}
    
    dist = [[] for _ in range(num_classes)]
    for label, count in class_counts.items():
        dist[label] += partition(class_ids[label], count)

    temp_dict = {i:[] for i in range(num_nodes)}
    for node in range(num_nodes):
        for i, label in enumerate(assigned_classes[node]):
            temp_dict[node] += [dist[label][0]]
            if len(dist[label]) != 0:
                dist[label].pop(0)
    dict_users= {i:list(chain.from_iterable(temp_dict[i])) for i in range(num_nodes)}
    for i in range(num_nodes):
        random.shuffle(dict_users[i])
        
    return dict_users