from DNN import *
import heapq
import numpy as np
from data_utils import DataSubset
import copy, gc

import torch
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset

class Nodes:
    """
    Generates node status and recording dictionaries
    """
    #idx, self.base_model, num_labels, in_channels, traindata, traindata_dist, testdata, testdata_dist, self.dataset, self.batch_size, node_n_nhood
    def __init__(self, node_idx: int, base_model: 'Net object', num_labels: int, in_channels: int, 
                 traindata, trg_dist:list, testdata, test_dist:list, dataset:str, batch_size:int,
                 node_neighborhood: list, lr = 0.01, wt_init = False, role = 'node'):
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
        self.role = role
        self.epochs = 0 # Time of creation -Assuming no learning has taken place. Necessary for LR scheduler
        self.lr = lr # LR employed by node
        
        # Dataset and data dist related
        self.trainset = trg_dist[self.idx]
        self.trainloader = DataLoader(DataSubset(traindata, trg_dist, self.idx), batch_size = batch_size)
        self.testset = test_dist[self.idx]
        self.testloader = DataLoader(DataSubset(testdata, test_dist, self.idx))
        self.base_model_selection(base_model, num_labels, in_channels, dataset, wt_init)
        
        # Recorders
        self.trgloss = []
        self.trgacc = []
        self.testloss = []
        self.testacc = []
        self.average_epochloss = 0
        
        # Appending self-idx to record CFL divergence
        # Divergence Targets
        div_targets = self.neighborhood
        self.divergence_dict = {node:[] for node in div_targets}
        self.divergence_conv_dict = {node:[] for node in div_targets}
        self.divergence_fc_dict = {node:[] for node in div_targets}  
        
    def base_model_selection(self, base_model, num_labels, in_channels, dataset, wt_init):
        # Same weight initialization
        self.model = copy.deepcopy(base_model)
#         if wt_init == True:
#             self.model.load_state_dict(base_mode.state_dict())
        self.opt = optim.SGD(self.model.parameters(), lr = self.lr) # , momentum = 0.9
#         lambda_sch = lambda epoch: 1 * epoch
#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda= lambda_sch)
 
    def local_update(self, num_epochs):
        # client_model, optimizer, scheduler, train_loader, record_loss, record_acc, epochs_done, num_epochs
        node_update(self.model, self.opt, self.trainloader, self.trgloss, self.trgacc, self.epochs, num_epochs)
        self.epochs += num_epochs
        print(f'Node{self.idx}-{self.epochs}', end = ', ', flush = True)
#         if len(self.trgloss) > 1:
#             print(f'Node {self.idx} : Delta Trgloss = {self.trgloss[-2] - self.trgloss[-1]:0.3f}', end = ",  ", flush = True)
#         else:
#             print(f'Node {self.idx}: Trgloss = {self.trgloss[-1]:0.3f}', end = ",  ")

    def node_test(self):
        test_loss, test_acc = test(self.model, self.testloader)
        self.testloss.append(test_loss)
        self.testacc.append(test_acc)
        print(f'Node{self.idx}: LR={self.opt.param_groups[0]["lr"]} Trg Loss= {self.trgloss[-1]:0.3f} Trg Acc= {self.trgacc[-1]} Test Acc= {self.testacc[-1]:0.3f}', end = ", ", flush = True)
         
    def scale_update(self, weightage):
        scale = {node:1.0 for node in self.neighborhood}
        return scale
            
    def aggregate_nodes(self, nodeset, agg_prop, scale:dict, cluster_set = None):
        # Choosing the #agg_count number of highest ranked nodes for aggregation
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
                    agg_targets.append(self.idx)
                except:
                    print(f'Agg_scope {agg_scope} does not conform to size of Cluster_Set {len(cluster_set)}')

        agg_model = aggregate(nodeset, agg_targets, scale)
        
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()
        
    def aggregate_random(self, nodeset, scale):
        target_id = self.idx
        while target_id == self.idx:
            target_id = random.sample(list(range(len(nodeset))), 1)[0]
        node_list = [self.idx, target_id]                     
        agg_model = aggregate(nodeset, node_list, scale)
#         self.model = copy.deepcopy(agg_model)
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()
        
    def cos_check(self, nodeset):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.cos_vals = {nhbr:None for nhbr in self.neighborhood}
        self_vec = vectorize_model(self.model, 'all')
        for neighbor in self.neighborhood:
            temp =vectorize_model(nodeset[neighbor].model, 'all')
            cos_sim = cos(self_vec, temp)
            self.cos_vals[neighbor] = cos_sim
    
    def aggregate_selective(self, nodeset):
        agg_full = []
        agg_conv = []
        for nhbr in self.neighborhood:
            if self.cos_vals[nhbr] > 0.5:
                agg_full.append(nhbr)
            elif 0.0 <= self.cos_vals[nhr] <= 0.5:
                agg_conv.appen(nhbr)
                
        agg_full.append(self.idx)
        agg_model = selective_aggregate(nodeset, agg_full, agg_conv, scale)
        self.model.load_state_dict(agg_model.state_dict())
        del agg_model
        gc.collect()
        

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
        self.model.load_state_dict(global_model.state_dict())

        for server in server_set:
            server.model.load_state_dict(self.model.state_dict())
            
        for node in nodeset:
            node.model.load_state_dict(self.model.state_dict())
        del global_model
        gc.collect()
        
    def aggregate_clusters(self, nodeset, assigned_nodes, scale, prop):
        nodelist = random.sample(assigned_nodes, int(prop*len(assigned_nodes)))
        server_agg_model = aggregate(nodeset, nodelist, scale)
        self.model.load_state_dict(server_agg_model.state_dict())
        del server_agg_model
        gc.collect()
        
def vectorize_model(model, key = 'all'):
    temp = []
    for layer in model.state_dict():
        if key != 'all': # Vectorize parts of model
            if 'weight' in layer and key in layer:
                temp.append(model.state_dict()[layer].view(-1))
        else: # Vectorize complete model
            if 'weight' in layer: 
                temp.append(model.state_dict()[layer].view(-1))
    x = torch.cat(temp)
    return x