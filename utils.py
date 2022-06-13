import random
import torch
import pickle

def constrained_sum(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur.
    """
    divider = []
    while 1 in divider or len(divider) == 0:
        dividers = sorted(random.sample(range(1, total), n - 1))
        divider = [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    return divider

def dataset_approve(dataset:'str'):
    if dataset == 'mnist': # Num labels will depend on the class in question
        location = '../data/'
        num_labels = 10
        in_ch = 1
    elif dataset == 'cifar':
        location = '../data/'
        num_labels = 10
        in_ch = 3
    elif dataset == 'fashion':
        location = '../data/'
        num_labels = 10
        in_ch = 1
    return location, num_labels, in_ch

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_mb = (param_size + buffer_size) / 1024**2
#     print('model size: {:.3f}MB'.format(model_mb))
    return model_mb


def save_file(location, modes, num_nodes):
    saved_set = {mode:{} for mode in modes}
    for mode in saved_set.keys():
        if mode != 'sgd':
            saved_set[mode] = {'avgtrgloss' : modes[mode].avgtrgloss,
                          'avgtrgacc' : modes[mode].avgtrgacc,
                          'avgtestloss' : modes[mode].avgtestloss,
                          'avgtestacc' : modes[mode].avgtestacc,
                          'cluster_trgloss' : modes[mode].cluster_trgloss,
                          'cluster_trgacc' : modes[mode].cluster_trgacc,
                          'cluster_testloss' : modes[mode].cluster_testloss,
                          'cluster_testacc' : modes[mode].cluster_testacc,
                          'nodetrgloss' : {node: modes[mode].nodeset[node].trgloss for node in range(num_nodes)},
                          'nodetrgacc' : {node:modes[mode].nodeset[node].trgacc for node in range(num_nodes)},
                          'nodetestloss' : {node:modes[mode].nodeset[node].testloss for node in range(num_nodes)},
                          'nodetestacc' : {node:modes[mode].nodeset[node].testacc for node in range(num_nodes)},
                          'divergence_dict' : {node:modes[mode].nodeset[node].divergence_dict for node in range(num_nodes)},
                          'divegence_cov_dict' : {node:modes[mode].nodeset[node].divergence_conv_dict for node in range(num_nodes)},
                          'divergence_fc_dict' : {node:modes[mode].nodeset[node].divergence_fc_dict for node in range(num_nodes)},
                          'neighborhood' : {node:modes[mode].nodeset[node].neighborhood for node in range(num_nodes)},
                          'ranked_nhood' : {node:modes[mode].nodeset[node].ranked_nhood for node in range(num_nodes)},
                          'node_degree' : {node:modes[mode].nodeset[node].degree for node in range(num_nodes)}
                          }
        elif mode == 'sgd':
            saved_set[mode] = {'avgtrgloss' : modes[mode].avgtrgloss,
                          'avgtrgacc' : modes[mode].avgtrgacc,
                          'avgtestloss' : modes[mode].avgtestloss,
                          'avgtestacc' : modes[mode].avgtestacc}
                
    with open(location, 'wb') as ffinal:
        pickle.dump(saved_set, ffinal) 

# Deprecated versions
#def generate_clusters(self):
#         cluster_set = []
#         cluster_sizes = []
#         # Ensure total nodes across clusters are equal to the total nodes
#         while sum(cluster_sizes) != self.num_nodes:
#             cluster_sizes = list(np.random.randint(self.min_size, self.max_size, size=self.num_clusters, dtype = int))
#         node_list = list(range(self.num_nodes))
#         random.shuffle(node_list)
#         for size in cluster_sizes:
#             temp = random.sample(node_list, size)
#             node_list = [item for item in node_list if item not in temp]
#             add_factor = np.random.randint(2, 6)
#             add = random.sample(list(range(self.num_nodes)),  add_factor)
#             for add_node in add:
#                 if add_node not in temp:
#                     temp.append(add_node)
#             cluster_set.append(temp)
#         self.cluster_set = cluster_set
#         print(f'The generated cluster set is {self.cluster_set}')


#def create_graph(self):
#         cluster_graph = nx.Graph()
#         for i in range(len(self.cluster_set)):
#             temp = nx.random_regular_graph(3, self.cluster_set[i], 4)
#             cluster_graph = nx.compose(cluster_graph, temp)
#             del temp
#         self.graph = cluster_graph


#Prev Model
#             if mode == 'hd2d': 
#                 # Create Hierarchical Servers
#                 modes[mode].form_serverset(environment.num_servers)
#                 # Assign Nodes
#                 node_list = list(range(num_nodes))
#                 for i in range(environment.num_servers):
#                     print(node_list)
#                     modes[mode].serverset[i].harchy_servers_allnodes(environment.cluster_ids[i], environment.cluster_set, node_list)
#                     node_list = [item for item in node_list if item not in modes[mode].serverset[i].node_ids]
                                 
#                     print(f'The nodes assigned to Server-{i} are {modes[mode].serverset[i].node_ids}')
#                 # Assign server list to Master Server
#                 modes[mode].serverset[-1].node_ids = list(range(environment.num_servers))
#                 print(f'The nodes assigned to Global Server are {modes[mode].serverset[-1].node_ids}')
            
#             if mode == 'hfl':
#                 modes[mode] = copy.deepcopy(modes['hd2d'])