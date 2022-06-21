import copy
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset

from DNN import *

class DataSubset(Dataset):
    """
    Takes the dataset, distribution list and node as arguments.
    """
    
    def __init__(self, dataset, datadist, node):
        self.dataset = dataset
        self.datadist = datadist
        self.indx = list(self.datadist[node])       
    
    def __len__(self):
        return len(self.indx)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.indx[item]]
#         image = self.dataset.data[item]
#         label = self.dataset.targets[item]
        return torch.tensor(image), torch.tensor(label)

def dataset_select(dataset, location, in_ch):
    """ 
    Select from MNIST, CIFAR-10 or FASHION-MNIST
    """
    ## MNIST
    if dataset == 'mnist' or dataset == 'fashion':
        if in_ch == 1:
           ### Choose transforms
            transform = transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))])
        elif in_ch == 3:
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Resize(224), 
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            transforms.Grayscale(num_output_channels = 3)]) #transforms.Lambda(lambda x: x.expand(3, -1, -1))

        if dataset == 'mnist':
            ## Create Train and Test Dataets
            traindata = torchvision.datasets.MNIST(root = location, train = True, download = True, transform = transform)
            testdata = torchvision.datasets.MNIST(root = location, train = False, download = True, transform = transform)
#         traindata = load_dataset(root = location, train = True, transform = transformations)
#         traindata.data = torch.unsqueeze(traindata.data, 1)
#         testdata = load_dataset(root = location, train = False, transform = transformations)
#         testdata.data = torch.unsqueeze(testdata.data, 1)

        elif dataset == 'fashion':
            # Download and load the training data
            traindata = datasets.FashionMNIST(root = location, download = True, train = True, transform = transform)
            testdata = datasets.FashionMNIST(root =location, download = True, train = False, transform = transform)
    
    ## CIFAR
    elif dataset == 'cifar':
        ### Choose transforms
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(224),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        traindata = torchvision.datasets.CIFAR10(root= location, train = True, download = True, transform = transform)
        testdata = torchvision.datasets.CIFAR10(root = location, train = False, download = True, transform = transform)
    
    
    else:
        raise NotImplementedError
      
    return traindata, testdata

def dict_creator(modes, dataset, num_labels, in_channels, num_nodes, num_rounds, wt_init):
        
    # Same weight initialization
    if wt_init == 'same':       
        same_wt_basemodel = Net(num_labels, in_channels, dataset)
        model_dict = {i:copy.deepcopy(same_wt_basemodel).cuda() for i in range(num_nodes)}
    elif wt_init == 'diff':
        model_dict = {i:Net(num_labels, in_channels, dataset).cuda() for i in range(num_nodes)}   
    
    recorder = {node:[] for node in range(num_nodes)}
    ## Model Dictionary for each of the Fed Learning Modes
    ## Model Dictionary Initialization

    mode_model_dict = {key:None for key in modes}
    mode_trgloss_dict = {key:None for key in modes}
    mode_testloss_dict = {key:None for key in modes}
    mode_avgloss_dict = {key:[] for key in modes}
    mode_acc_dict = {key:None for key in modes}
    mode_avgacc_dict = {key:[] for key in modes}
    
    basemodel_keys = model_dict[0].state_dict().keys()
    layer_dict = {layer:[] for layer in basemodel_keys}
    nodelayer_dict = {node:copy.deepcopy(layer_dict) for node in range(num_nodes)}   
    divergence_dict = {mode:copy.deepcopy(nodelayer_dict) for mode in modes if mode != 'SGD'}  
    
    # Create separate copies for each mode
    for mode in modes:
        if mode != 'SGD':
            mode_model_dict[mode] = copy.deepcopy(model_dict)
            mode_trgloss_dict[mode] = copy.deepcopy(recorder)
            mode_testloss_dict[mode] = copy.deepcopy(recorder)
            mode_acc_dict[mode] = copy.deepcopy(recorder)
        elif mode == 'SGD':
            mode_model_dict[mode] = copy.deepcopy(same_wt_basemodel).cuda()
            mode_trgloss_dict[mode] = []
            mode_testloss_dict[mode] = []
            mode_acc_dict[mode] = []
    return mode_model_dict, mode_trgloss_dict, mode_testloss_dict, mode_avgloss_dict, mode_acc_dict, mode_avgacc_dict, divergence_dict

        
