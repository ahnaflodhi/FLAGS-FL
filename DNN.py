import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import random
import copy
import heapq
import sys, gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from utils import optimizer_to, scheduler_to

class Net(nn.Module):
    def __init__(self, num_classes, in_ch, dataset):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        if dataset == 'mnist' or dataset == 'fashion':
            self.fc1 = nn.Linear(2000, 1000)
            self.fc2 = nn.Linear(1000, num_classes)
        elif dataset == 'cifar':
            self.fc1 = nn.Linear(2880, 1440)
            self.fc2 = nn.Linear(1440, num_classes)
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_ch, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.2)
#         self.dropout2 = nn.Dropout(0.1)
#         if dataset == 'mnist' or dataset == 'fashion':
#             self.fc1 = nn.Linear(9216, 128)
#             self.fc2 = nn.Linear(128, num_classes)
#         elif dataset == 'cifar':
#             self.fc1 = nn.Linear(12544, 128)
#             self.fc2 = nn.Linear(128, num_classes)
            
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    @staticmethod
    def add_noise(model, mean = [0.0], std = [0.01]):
        norm_dist = torch.distributions.Normal(loc = torch.tensor(mean), scale = torch.tensor(std))
        for layer in model.state_dict():
            if 'weight' in layer:
                x = model.state_dict()[layer]
                t = norm_dist.sample((x.view(-1).size())).reshape(x.size()).cuda()
                model.state_dict()[layer].add_(t)
        return model
   
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
# self.model, self.opt, self.trainset, self.trainloader, self.trgloss, self.trgacc, num_epochs   

def node_update(client_model, optimizer, train_loader, record_loss, record_acc, epochs_done, num_epochs):
#     optimizer_to(optimizer, 'cuda')
#     scheduler_to(scheduler, 'cuda')
    client_model.train()
    for epoch in range(num_epochs):
#         epoch_loss = 0.0
        batch_loss = []
        correct_state = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            correct = 0
            data = data.float()
            data, targets = data.cuda(), targets.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.cross_entropy(output, targets)
            _ , output = torch.max(output.data, 1)
            loss.backward()
            optimizer.step()
            correct += (output == targets).float().sum() / output.shape[0]
            batch_loss.append(loss.item())
            correct_state.append(correct.item())
#             if batch_idx % 100 == 0:    # print every 100 mini-batches
#                 print('[%d, %5d] loss-acc: %.3f - %.3f' %(epoch+1, batch_idx+1, sum(batch_loss)/len(batch_loss), sum(correct_state)/len(correct_state)))
#         if epochs_done > 0 and epochs_done % 20 == 0: # Reduce LR after this many epocs.
#             scheduler.step()
            
        epoch_loss = sum(batch_loss) / len(batch_loss)
        epoch_acc = sum(correct_state) / len(correct_state)
        record_loss.append(round(epoch_loss, 3))
        record_acc.append(round(epoch_acc,3))
#     del data, targets, batch_loss
#     del loss, output, correct_state
#     del epoch_loss, epoch_acc
#     gc.collect()
    
def aggregate(model_list, node_list, scale, noise = False):
    agg_model = copy.deepcopy(model_list[0].model)
    ref_dict = copy.deepcopy(agg_model.state_dict())
    if noise == True: # Create copies so that original models are not corrupted. Only received ones become noisy
        models = [copy.deepcopy(model_list[node].model) for node in node_list]
        models = [Net.add_noise(model) for model in models]
        for k in ref_dict.keys():
            ref_dict[k] = torch.stack([models[i].state_dict()[k].float() for i, _ in enumerate(node_list)], 0).mean(0)
        del models
        
    else:
        for k in ref_dict.keys():
            ref_dict[k] = torch.stack([model_list[i].model.state_dict()[k].float() for i, _ in enumerate(node_list)], 0).mean(0)
    gc.collect()
#         for k in ref_dict.keys():
#             ref_dict[k] = torch.stack([torch.mul(models[i].state_dict()[k].float(), scale[node]) for i, node in enumerate(node_list)], 0).mean(0)
    
    agg_model.load_state_dict(ref_dict)
    gc.collect()
    return agg_model

def model_checker(model1, model2):
    models_differ = 0
    for modeldata1, modeldata2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(modeldata1[1], modeldata2[1]):
            pass
        else:
            models_differ += 1
            if (modeldata1[0] ==  modeldata2[0]):
                print("Mismatch at ", modeldata1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

def extract_weights(model, add_noise = True):
    weights = {}
    for key in model.state_dict():
        if 'weight' not in key:
            continue
        weights[key] = model.state_dict()[key]
    return weights

def calculate_divergence(modes, main_model_dict, cluster_set, num_nodes, divergence_results):
    centr_fed_ref = np.random.randint(0, num_nodes)
    for mode in modes:
        basemodel_keys = main_model_dict[mode][0].state_dict().keys()
        break
     # Structure of Dictionary   
    # divergence_results {mode: {node:{layer:[divergence for each round]}}       
                                      
    ref_model = main_model_dict['SGD']
    ref_weight = extract_weights(ref_model)
    
    for mode in modes:
        if mode != 'SGD':
            for target_node in range(num_nodes):
                target_model = main_model_dict[mode][target_node].cuda()
                target_weight = extract_weights(target_model)
                for layer in ref_weight.keys():                          
                    divergence_results[mode][target_node][layer].append(torch.linalg.norm(ref_weight[layer] - target_weight[layer]))
    return  divergence_results


def clustering_divergence(model_dict, cluster_graph, num_nodes):
    divergence_results = []
    neighborhood = []
    basemodel_keys = model_dict[0].state_dict().keys()
    div_recorder = {}
    div_recorder_conv = {}
    div_recorder_fc = {}
    
    for node in range(num_nodes):
        temp = [neighbor for neighbor in cluster_graph.neighbors(node)]
        neighborhood.append(temp)
        div_recorder[node] = {neighbor:None for neighbor in temp}
        div_recorder_conv[node] = {neighbor:None for neighbor in temp}
        div_recorder_fc[node] = {neighbor:None for neighbor in temp}
        
        
    for ref_node, neighbor_nodes in enumerate(neighborhood):
        for neighbor in neighbor_nodes:
            total_diff = 0
            conv_diff = 0
            fc_diff = 0
            for layer in model_dict[ref_node].state_dict():
                if 'weight' not in layer:
                    continue
                diff = torch.linalg.norm(model_dict[ref_node].state_dict()[layer] - model_dict[neighbor].state_dict()[layer])
                total_diff += diff.item()
                if 'conv' in layer:
                    conv_diff += diff.item()
                elif 'fc' in layer:
                    fc_diff += diff.item()
                
            div_recorder[ref_node][neighbor] = total_diff
            div_recorder_conv[ref_node][neighbor] = conv_diff
            div_recorder_fc[ref_node][neighbor] = fc_diff
    
     #Normalize
    div_recorder = normalize_div(div_recorder)
    div_recorder_conv = normalize_div(div_recorder_conv)
    div_recorder_fc = normalize_div(div_recorder_fc)
            
    return div_recorder, div_recorder_conv, div_recorder_fc

def normalize_div(div_dict):
    for node in range(len(div_dict)):
        temp = []
        print(div_dict[node])
        for _ , val in div_dict[node].items():
            temp.append(val)
            norm_factor = np.linalg.norm(temp)
            
        for neighbor, _ in div_dict[node].items():
            div_dict[node][neighbor] = div_dict[node][neighbor] / norm_factor
    return div_dict

def revise_neighborhood(div_dict, n, sort_type = 'min'):
    revised_neighborhood = []
    for node in range(len(div_dict)):
        if sort_type == 'min':
            temp = heapq.nsmallest(n, div_dict[node].items() , key=lambda i: i[1])
        elif sort_type == 'max':
            temp = heapq.nlargest(n, div_dict[node].items() , key=lambda i: i[1])
        temp_nodes = [max_node[0] for max_node in temp]
        revised_neighborhood.append(temp_nodes)
    return revised_neighborhood
    
                     