from DNN import *
import heapq
import numpy as np
from data_utils import DataSubset
import copy

import torch
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset

class Nodes:
    """
    Generates node status and recording dictionaries
    """
    
    def __init__(self, node_idx: int, base_model: 'Net object', num_labels: int, in_channels: int, 
                 traindata, trg_dist:list, testdata, test_dist:list, dataset:str, batch_size:int,
                 node_neighborhood: list, network_weights: list, lr = 0.1, wt_init = False, role = 'node'):
        """
        Creates the Node Object.
        Contains methods for individual nodes to perform.
        Requires neighborhood information for each node.
        """
        #Node properties
        self.idx = node_idx
        self.batch_size = batch_size
        self.neighborhood = node_neighborhood
        self.ranked_nhood = node_neighborhood
        self.degree = len(self.neighborhood)
        self.weights = network_weights[self.idx]
        self.role = role
        self.epochs = 0 # Time of creation -Assuming no learning has taken place. Necessary for LR scheduler
        
        # Dataset and data dist related
        self.trainset = trg_dist[self.idx]
        self.trainloader = DataLoader(DataSubset(traindata, trg_dist, self.idx), batch_size = batch_size)
        self.testset = test_dist[self.idx]
        self.testloader = DataLoader(DataSubset(testdata, test_dist, self.idx))
        self.base_model_selection(base_model, num_labels, in_channels, dataset, wt_init, lr)
        
        # Recorders
        self.trgloss = []
        self.trgacc = []
        self.testloss = []
        self.testacc = []
        
        # Appending self-idx to record CFL divergence
        # Divergence Targets
        div_targets = self.neighborhood
        self.divergence_dict = {node:[] for node in div_targets}
        self.divergence_conv_dict = {node:[] for node in div_targets}
        self.divergence_fc_dict = {node:[] for node in div_targets}  
        
    def base_model_selection(self, base_model, num_labels, in_channels, dataset, wt_init, lr):
        # Same weight initialization
        if wt_init == True:
            self.model = copy.deepcopy(base_model)
        else:
            self.model = Net(num_labels, in_channels, dataset)
        self.opt = optim.SGD(self.model.parameters(), lr = lr, momentum = 0.9)
        
    def local_update(self, num_epochs):
        node_update(self.model, self.opt, self.trainloader, self.trgloss, self.trgacc, self.epochs, num_epochs)
#         if len(self.trgloss) > 1:
#             print(f'Node {self.idx} : Delta Trgloss = {self.trgloss[-2] - self.trgloss[-1]:0.3f}', end = ",  ", flush = True)
#         else:
#             print(f'Node {self.idx}: Trgloss = {self.trgloss[-1]:0.3f}', end = ",  ")
    
    def node_test(self):
        test_loss, test_acc = test(self.model, self.testloader)
        self.testloss.append(test_loss)
        self.testacc.append(test_acc)
        print(f'Node {self.idx}: Trg Loss = {self.trgloss[-1]:0.3f} Trg Acc: {self.trgacc[-1]}  Test Acc : {self.testacc[-1]}', end = ",  ", flush = True)
#         print(f'Accuracy for node{self.idx} is {test_acc:0.5f}')          
    def neighborhood_divergence(self, nodeset, cfl_model,  div_metric = 'L2', div_mode ='cfl_div', normalize = False):
        div_dict = {node:None for node in self.neighborhood}
        total_div_dict = copy.deepcopy(div_dict)
        conv_div_dict = copy.deepcopy(div_dict)
        fc_div_dict = copy.deepcopy(div_dict)
        
        for nhbr_node in self.neighborhood:
            totaldiv, convdiv, fcdiv = self.internode_divergence(cfl_model, nodeset[nhbr_node].model, div_metric, div_mode)
#             print(total_div, conv_div, fc_div)
#             print(self.divergence_dict)
            self.divergence_dict[nhbr_node].append(totaldiv)
            self.divergence_conv_dict[nhbr_node].append(convdiv)
            self.divergence_fc_dict[nhbr_node].append(fcdiv)
            
        if normalize == True:
            self.normalize_divergence()
    
    def internode_divergence(self, cfl_model, target_model, div_metric, div_mode):
        total_div = 0
        conv_div = 0
        fc_div  = 0
        
        if div_mode == 'internode':        
            ref_wt = extract_weights(self.model)
            target_wt = extract_weights(target_model)
        elif div_mode == 'cfl_div':
            ref_wt = extract_weights(cfl_model)
            target_wt = extract_weights(target_model)
            
        for layer in ref_wt.keys():
            if 'weight' not in layer:
                continue
            if div_metric == 'L2':
                diff = torch.linalg.norm(ref_wt[layer] - target_wt[layer]).item()
            total_div += diff
            if 'conv' in layer:
                conv_div += diff
            if 'fc' in layer:
                fc_div += diff
                             
        return total_div, conv_div, fc_div
      
    def normalize_divergence(self):
        total_div_factor = sum([self.divergence_dict[nhbr][-1] for nhbr in self.divergence_dict.keys()])
        conv_div_factor = sum([self.divergence_conv_dict[nhbr][-1] for nhbr in self.divergence_conv_dict.keys()])
        fc_div_factor = sum([self.divergence_fc_dict[nhbr][-1] for nhbr in self.divergence_fc_dict.keys()])
        
        for nhbr in self.divergence_dict.keys():
            self.divergence_dict[nhbr][-1] = self.divergence_dict[nhbr][-1] / total_div_factor
            self.divergence_conv_dict[nhbr][-1] = self.divergence_conv_dict[nhbr][-1] / conv_div_factor
            self.divergence_fc_dict[nhbr][-1] = self.divergence_fc_dict[nhbr][-1] / fc_div_factor
    
    def nhood_ranking(self, rnd, mode_name, sort_crit = 'total', sort_scope = 1, sort_type = 'min'):
        if mode_name == 'd2d_up':
            sort_type = 'min'
        elif mode_name == 'd2d_down':
            sort_type =  'max'
        elif mode_name == 'd2d_main':
            sort_type = None
            
        if sort_crit == 'total':
            self.apply_ranking(self.divergence_dict, rnd, sort_scope, sort_type)
        elif sort_crit == 'conv':
            self.apply_ranking(self.divergence_conv_dict, rnd, sort_scope, sort_type)
        elif sort_crit == 'fc':
            self.apply_ranking(self.divergence_fc_dict, rnd, sort_scope, sort_type)
    
    def apply_ranking(self, target, rnd, sort_scope, sort_type):
        # Target is the metric (divergence, KL, WS) to apply ranking on.
        if rnd == 0 or sort_type is None:
            self.ranked_nhood = self.neighborhood           
        else:
            prev_performance = {neighbor:sum(divergence[-sort_scope:]) for neighbor, divergence in target.items()}
            if sort_type == 'min':
#                     sorted_nhood ={k: v for k, v in sorted(prev_performance.items(), key=lambda item: item[1])}
                sorted_nhood = heapq.nsmallest(len(target), prev_performance.items(), key = lambda i:i[1])
            elif sort_type == 'max':
                sorted_nhood = heapq.nlargest(len(target), prev_performance.items(), key = lambda i:i[1])
            self.ranked_nhood = [nhbr for nhbr, _ in sorted_nhood]

    def scale_update(self, weightage):
#         # Aggregation Weights
#         if weightage == 'equal':
#             # Same weights applied to all aggregation
#             scale = {node:1.0 for node in self.neighborhood}
#         elif weightage == 'proportional':
#             # Divergence-based weights applied to respective models.
#             scale = {node:self.divergence_dict[node][-1] for node in self.neighborhood}
        scale = {node:1.0 for node in self.neighborhood}
        return scale
            
    def aggregate_nodes(self, nodeset, agg_prop, weightage, cluster_set = None):
        #Choosing the #agg_count number of highest ranked nodes for aggregation
        # If Node aggregating Nhood
        if cluster_set == None:
            agg_scope = int(np.floor(agg_prop * len(self.ranked_nhood)))
            if agg_scope >= 1 and agg_scope <= len(self.ranked_nhood):
                try:
                    agg_targets = self.ranked_nhood[:agg_scope]
                    agg_targets.append(self.idx)
                except:
                    print(f'Agg_scope:{agg_scope} does not conform to neighborhood {len(self.ranked_nhood)}')
        # If CH aggregating Cluster    
        else:
            agg_scope = int(np.floor(agg_prop * len(cluster_set)))
            if agg_scope >= 1 and agg_scope <= len(cluster_set):
                try:
                    # No need to add self index since cluster-head id already included in cluster-set
                    agg_targets = random.sample(cluster_set, agg_scope)
                except:
                    print(f'Agg_scope {agg_scope} does not conform to size of Cluster_Set {len(cluster_set)}')
        scale = self.scale_update(weightage)
        agg_model = aggregate(nodeset, agg_targets, scale)
        self.model = copy.deepcopy(agg_model)
#         self.model.load_state_dict(agg_model.state_dict())
        
    def aggregate_random(self, nodeset, weightage):
        target_id = self.idx
        while target_id == self.idx:
            target_id = random.sample(list(range(len(nodeset))), 1)[0]
        node_list = [self.idx, target_id]
        if weightage  == 'equal':
            scale = {node:1.0 for node in node_list}
        elif weightage == 'proportional':
            scale = {node:1.0 for node in node_list}
                     
        agg_model = aggregate(nodeset, node_list, scale)
        self.model = copy.deepcopy(agg_model)
#         self.model.load_state_dict(agg_model.state_dict())
        
        
class Servers:
    def __init__(self, server_id, model, records = False):
        self.idx = server_id
        self.model = copy.deepcopy(model)
        if records == True:
            self.avgtrgloss = []
            self.avgtrgacc = []
            self.avgtestloss = []
            self.avgtestacc =[]
             
    def aggregate_servers(self, server_set, nodeset):
        scale = {server_id:1.0 for server_id in range(len(server_set)) }
        global_model = aggregate(server_set, list(range(len(server_set))), scale)
        self.model = copy.deepcopy(global_model)
#         self.model.load_state_dict(global_model.state_dict())

        for server in server_set:
            server.model = copy.deepcopy(self.model)
#             server.model.load_state_dict(self.model.state_dict())
            
        for node in nodeset:
            node.model = copy.deepcopy(self.model)
#             node.model.load_state_dict(self.model.state_dict())
        
    def aggregate_clusters(self, nodeset, assigned_nodes, prop):
        nodelist = random.sample(assigned_nodes, int(prop*len(assigned_nodes)))
        scale = {node:1.0 for node in nodelist}
        server_agg_model = aggregate(nodeset, nodelist, scale)
        self.model = copy.deepcopy(server_agg_model)
#         self.model.load_state_dict(server_agg_model.state_dict())