import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# #The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys, argparse
import pickle
import time
import gc
from get_args import arg_parser
from utils import dataset_approve, save_file, model_size
from data_utils import * # Returns the train and test sets for the chosen dataset; dataset_select and class DataSubset
from data_dist import * # (Returns the dictionary of nodes/data partitions for both iid and nidd) )
from DNN import * # (Returns Network, client update, aggregate)
from env_sysmodel import system_model, FL_Modes
from devices import Nodes, Servers

args = arg_parser()
dataset = args.d
batch_size = args.b
nodes = args.n
clusters = args.c
epochs = args.e
rounds = args.r
overlap_factor = args.o
shards =args.s
dist_mode = args.dist
test_batch_size = args.t
prop = args.prop
agg_prop = args.aggprop
servers = args.ser

 
modes_list = {'d2d':None, 'chd2d':None, 'hch_d2d': None, 'gossip':None, 'hgossip':None, 'cfl': None, 'sgd' : None}

def D2DFL(dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_servers, num_rounds, 
          num_epochs, shard_size, overlap, dist, prop, agg_prop):
    
    # Step 1: Define parameters for the environment, dataset and dataset distribution
    starttime = time.strftime("%Y%m%d-%H%M")
    location, num_labels, in_ch = dataset_approve(dataset)    
    base_model = Net(num_labels, in_ch, dataset)

    
    #### Step 2: Import Dataset partitioned into train and testsets
    # Call data_select from data_utils
    traindata, testdata = dataset_select(dataset, location)

    #### Step 3: Divide data among the nodes according to the distribution IID or non-IID
    # Call data_iid/ data_noniid from data_dist
    if dist == 'iid':
        train_dist = data_iid(traindata, num_labels, num_nodes)
    elif dist == 'niid':
        train_dist = data_noniid(traindata, num_nodes, shard_size)
    elif dist == 'niid1':
        skew = 1
        train_dist = niid_skew_dist(traindata, num_labels, num_nodes, skew, shard_size)
    elif dist == 'niid2':
        skew = 2
        train_dist = niid_skew_dist(traindata, num_labels, num_nodes, skew, shard_size)
    
    # Uniform Test distribution for each node. The testing may be carried out on the entire datset
    test_dist = data_iid(testdata, num_labels, num_nodes)
    
    # Step 4: Create Environment
    env = system_model(num_nodes, num_clusters, num_servers)
    
    # Create Base Parameter Dictionary for Modes
    base_params = { 'dataset' : dataset, 'num_epochs' : num_epochs, 'num_rounds' : num_rounds, 
                   'num_nodes' : num_nodes, 'dist' : dist, 'base_model' : base_model,'num_labels' : num_labels, 
                   'in_channels' : in_ch, 'traindata' : traindata, 'traindata_dist' : train_dist, 
                   'testdata' : testdata, 'testdata_dist' : test_dist, 'batch_size' : batch_size,
                   'nhood' : env.neighborhood_map, 'env_Lp' : env.Lp, 'num_clusters' : num_clusters,
                   'num_servers': env.num_servers}
    
    d2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    hd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    hfl_flags = {'d2d_agg_flg' : False, 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    chd2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': True, 'hserver_agg_flg': False, 'inter_ch_agg_flg': True}
    hch_d2d_flags = {'d2d_agg_flg' : 'D2D', 'ch_agg_flg': True, 'hserver_agg_flg': True, 'inter_ch_agg_flg': True}
    gossip_flg = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    hgossip_flg = {'d2d_agg_flg' : 'Random', 'ch_agg_flg': False, 'hserver_agg_flg': True, 'inter_ch_agg_flg': False}
    cfl_flg = {'d2d_agg_flg' : 'CServer', 'ch_agg_flg': False, 'hserver_agg_flg': False, 'inter_ch_agg_flg': False}
    
    flag_dict = {'d2d': d2d_flags, 'hd2d': hd2d_flags, 'hfl': hfl_flags, 'chd2d':chd2d_flags, 'hch_d2d': hch_d2d_flags, 
                 'gossip':gossip_flg, 'hgossip':hgossip_flg, 'cfl':cfl_flg, 'sgd':None}
    
    # Step-5: Create Modes and combine mode params and special flags for all modes under mode_params
    mode_params = {mode:None for mode in modes.keys()}
    for mode in modes.keys():
        if flag_dict[mode] != None:
            mode_params[mode] = {**base_params, **flag_dict[mode]}
        else:
            mode_params[mode] = base_params
        mode_params[mode]['name'] = mode
        
    modelist = [mode for mode in modes.keys()]
    for mode in modelist:
        file_args = {'folder': None, 'status': None, 'flmode': None, 'modename': mode, 'dataset':dataset, 'dist':dist, 'num_nodes':num_nodes, 'num_clusters':num_clusters,
                                 'num_epochs':num_epochs, 'num_rounds': num_rounds, 'starttime':starttime}
        if mode != 'sgd':
            # Creates Nodeset and other attributes for each mode in modes
            modes[mode] = FL_Modes(**mode_params[mode])
            # Check Hierarchical Aggregation Flag
            if modes[mode].hserver_agg_flg == True:
            # Create Hierarchical Servers
                modes[mode].form_serverset(env.num_servers, num_labels, in_ch, dataset)
            
            # Start Federation Protocol
            for rnd in range(num_rounds):
                ### Move Mode-models to cuda
                for node in modes[mode].nodeset:
                    node.model.to('cuda')
                if hasattr(modes[mode], 'serverset'):
                    for server in modes[mode].serverset:
                        server.model.to('cuda')
                modes[mode].cfl_model.to('cuda')
#                 print(f'The Cuda Summary before update {torch.cuda.memory_summary()}')
                # Initiate Local Training on models
                modes[mode].update_round()
#                 print(f'The Cuda Summary after update {torch.cuda.memory_summary()}')

                # Perform Testing on Locally trained/fine-tuned models
                modes[mode].test_round(env.cluster_set)
#                 print(f'The Cuda Summary before test {torch.cuda.memory_summary()}')

                # Share models with neighbors
                # Add noise / Share partials / 

                # Perform Neighborhood analysis and update weights assigned to neighbors
                modes[mode].ranking_round(rnd, mode)
#                 print(f'The Cuda Summary after ranking {torch.cuda.memory_summary()}')

                #4-Aggregate from neighborhood  using the weights obtained in the previous step
                print(f'Starting Local Aggregation in round{rnd} for mode {mode}')
                if modes[mode].d2d_agg_flg == 'D2D':
                    modes[mode].nhood_aggregate_round(agg_prop)

                elif modes[mode].d2d_agg_flg == 'Random':
                    modes[mode].random_aggregate_round()

                elif modes[mode].d2d_agg_flg == 'CServer':
                    modes[mode].cfl_aggregate_round(prop)
#                 print(f'The Cuda Summary after aggregation {torch.cuda.memory_summary()}')

                # 5- Cluster operations: 
                if modes[mode].ch_agg_flg == True:
                    print(f'Entering Cluster Head Aggregation for mode-{mode} in round-{rnd}')
                    for i in range(env.num_clusters):
                        modes[mode].clshead_aggregate_round(env.cluster_heads[i], env.cluster_set[i], agg_prop)

                if modes[mode].inter_ch_agg_flg == True:
                    modes[mode].inter_ch_aggregate_round(env.cluster_heads)
#                 print(f'The Cuda Summary after cluster operation {torch.cuda.memory_summary()}')

                # Should not be executed for Clustered D2D-FL
                if modes[mode].hserver_agg_flg == True: 
                    print(f'Entering Hierarchical Aggregation for mode-{mode} in round-{rnd}')
                    assigned_nodes = []
                    for i in range(env.num_servers):
                        for cluster_id in env.server_groups[i]:
                            assigned_nodes += env.cluster_set[cluster_id] 
                        modes[mode].serverset[i].aggregate_clusters(modes[mode].nodeset, assigned_nodes, prop)
#                     print(f'The Cuda Summary after Hserver aggregation {torch.cuda.memory_summary()}')

                    #Final Server Aggregation
                    modes[mode].serverset[-1].aggregate_servers(modes[mode].serverset[:-1], modes[mode].nodeset)

                # Interim Record
                if rnd % 5 == 0:
                    file_args['folder'] = './'
                    file_args['status'] = 'inter'
                    file_args['flmode'] = modes[mode]
                    save_file(**file_args)
             
            file_args['folder'] = './Results'
            file_args['status'] = 'Final'
            file_args['flmode'] = modes[mode]
            print(f'Printing mode from Main {modes[mode]}')
            save_file(**file_args)
            
            del modes[mode]            
            gc.collect()
                     
#             for node in modes[mode].nodeset:
#                 model_sizes += model_size(node.model)
#                 node.model.to('cpu')

#             if hasattr(modes[mode], 'serverset'):
#                 for server in modes[mode].serverset:
#                     model_sizes += model_size(server.model)
#                     server.model.to('cpu')
#             model_sizes = model_size(modes[mode].cfl_model)
#             modes[mode].cfl_model.to('cpu')
#                 print(f'The Cuda Summary after moving models to CPU {torch.cuda.memory_summary()}')
            torch.cuda.empty_cache()

        elif mode == 'sgd':
            modes[mode] = Servers(0, base_model, records = True)
            sgd_optim = optim.SGD(modes[mode].model.parameters(), lr = 0.01, momentum = 0.9)
            sgd_trainloader = DataLoader(traindata, batch_size = 32)
            sgd_testloader =  DataLoader(testdata)
            for rnd in range(num_rounds * num_epochs):
                node_update(modes[mode].model.cuda(), sgd_optim, sgd_trainloader, modes[mode].avgtrgloss,
                            modes[mode].avgtrgacc, 0, num_epochs)
                loss, acc = test(modes[mode].model, sgd_testloader)
                modes[mode].avgtestloss.append(loss)
                modes[mode].avgtestacc.append(acc)
            
            file_args['folder'] = './Results'
            file_args['status'] = 'Final'
            file_args['flmode'] = modes[mode]
            save_file(**file_args)
            
            del modes[mode]
            gc.collect()
            
## Main Function
if __name__ == "__main__":
#     dataset, batch_size, test_batch_size, modes, num_nodes, num_clusters, num_rounds, num_epochs, shard_size, overlap, dist
    mode_state = D2DFL(dataset, batch_size, test_batch_size, modes_list,  nodes, clusters, servers, rounds, epochs, shards, overlap_factor, dist_mode, prop, agg_prop)