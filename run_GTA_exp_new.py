#!/usr/bin/env python
# coding: utf-8

# In[1]: 


#import imp
import time
import sys
import argparse
from unittest import result
import numpy as np
import torch
from exp_method import Exp_worker
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI


# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated,prune_with_exp,prune_with_exp_sage_v4
import scipy.sparse as sp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='Pubmed', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2','Yelp'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--vs_size', type=int, default=40,
                    help="ratio of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none','exp'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.5,
                    help="Threshold of prunning edges")
parser.add_argument('--homo_loss_weight', type=float, default=0,
                    help="Weight of optimize similarity loss")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=3,
                    help="Threshold of prunning edges")
parser.add_argument('--vs_ratio', type=float, default=0.005,
                    help="ratio of poisoning nodes relative to the full graph")
# args = parser.parse_args()
#explain setting
parser.add_argument('--cut_num', type=int, default=2,
                    help="cut number in exp defense")
parser.add_argument('--use_target_node', type=int, default=0,
                    help="wheather remove target node from idx_atk")
parser.add_argument('--defense_exp_test', type=int, default=0,
                    help="test mode for learn clean acc")
parser.add_argument('--max_atk_num',type=int,default=400,
                    help="size of atk_num")
parser.add_argument('--exp_alg',type=str,default='Captum',
                    choices=['Captum','Attention','GNNExp','GraphMask'],
                    help="explain algorithm")
parser.add_argument('--exp_mode',type=str,default='none',choices=['none','Simple'],
                    help="size of atk_num")
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                    transform=transform)
elif(args.dataset == 'Reddit2'):
    dataset = Reddit2(root='./data/Reddit2/', \
                    transform=transform)
elif(args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 

data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
# we build our own train test split 
#%% 
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
if idx_atk.shape[0]>args.max_atk_num:
    idx_atk=idx_atk[:args.max_atk_num]
from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


# In[9]:
from models.GTA import Backdoor
import heuristic_selection as hs

# from kmeans_pytorch import kmeans, kmeans_predict
if args.use_target_node==0:
    idx_atk=idx_atk[data.y[idx_atk]!=args.target_class]
    print('dont use_target')

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
#size = args.vs_size #int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
size =int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
print("#Attach Nodes:{}".format(size))
# here is randomly select poison nodes from unlabeled nodes
from models.construct import model_construct
from torch_geometric.utils  import k_hop_subgraph
import time
models = ['GCN','GAT','GraphSage']
# models = ['GCN']
models_asr=[]
models_acc=[]
for test_model in models:
    args.test_model = test_model
    result_asr = []
    result_acc = [] 
    result_asr_p = []
    result_acc_p = []
    ACC_seeds=[]
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000,size=5)
    for seed in seeds:
        print(args)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        args.seed = seed
        if(args.selection_method == 'none'):
            idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
        elif(args.selection_method == 'cluster'):
            idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif(args.selection_method == 'cluster_degree'):
            idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
            idx_attach = torch.LongTensor(idx_attach).to(device)

        # In[10]:
        # train trigger generator 
        st=time.time()
        model = Backdoor(args,device)
        model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
        ed=time.time()
        print('time=',ed-st)
        print('*'*20)
        poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()
        
        if(args.defense_mode == 'prune'):
            poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
            bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
        elif(args.defense_mode == 'isolate'):
            poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
            bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
            bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
        else:
            bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
        print("precent of left attach nodes: {:.3f}"\
            .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))
        #%%


        test_model = model_construct(args,args.test_model,data,device).to(device) 
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
        test_model_p = model_construct(args,args.test_model,data,device).to(device) 
        test_model_p.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)

        output = test_model(poison_x,poison_edge_index,poison_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
        output_p = test_model_p(poison_x,poison_edge_index,poison_edge_weights)
        train_attach_rate_p = (output_p.argmax(dim=1)[idx_attach]==args.target_class).float().mean()

        print("target class rate on Vs: {:.4f}".format(train_attach_rate))
        print("Para target class rate on Vs: {:.4f}".format(train_attach_rate_p))
        if args.defense_mode == 'exp':
            exp_test=Exp_worker(test_model,data.y,algorithm=args.exp_alg)###

        torch.cuda.empty_cache()
        #%%
        induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
        # test_model = test_model.cpu()
        clean_acc_p = test_model_p.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

        print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
        print("Para accuracy on clean test nodes: {:.4f}".format(clean_acc))
        if args.defense_exp_test==1 and args.defense_mode=='none':
            overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
            print('Model=',args.test_model)
            total_rate=0
            flag=True
            total_rate_ori=0
            for i,idx in enumerate(idx_clean_test):#取一个子集做测试 非中毒条件下是否影响准确性
                sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                ori_node_idx = sub_induct_nodeset[sub_mapping]
                relabeled_node_idx = sub_mapping
                sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                induct_x, induct_edge_index,induct_edge_weights=poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights
                with torch.no_grad():
                    output_ori = test_model(induct_x,induct_edge_index,induct_edge_weights)
                    train_attach_rate_ori = (output_ori.argmax(dim=1)[relabeled_node_idx]==data.y[idx]).float().mean()
                    # induct_edge_index,induct_edge_weights,xl=prune_with_exp_sage_v4(args,
                    #                                                         test_model,induct_edge_index,
                    #                                                         induct_edge_weights,induct_x,
                    #                                                         device,data.y,sub_mapping,)
                    args.defense_mode='prune'
                    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device,False)
                    
                    output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                    train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==data.y[idx]).float().mean()
                    total_rate+=train_attach_rate
                    total_rate_ori+=train_attach_rate_ori
                    if i%50==0:
                        
                        print(i,'  ',i/min(len(idx_clean_test),2000),' ',total_rate/(i+1))
                        print(i,'  ',i/min(len(idx_clean_test),2000),' ',total_rate_ori/(i+1))
                    if i==min(len(idx_clean_test),2000):break
            print('clean defense test complete')
            print('ACC_DEFENSE={:.4f}'.format(total_rate/(min(len(idx_clean_test),2000)+1)))
            print('ACC_DEFENSE_ori={:.4f}'.format(total_rate_ori/(min(len(idx_clean_test),2000)+1)))
            #print('ACC_DEFENSE={:.4f}'.format(total_rate/idx_clean_test.shape[0]))
            sys.exit()
        if(args.evaluate_mode == '1by1'):
            overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
            asr = 0
            ACC=0
            flip_asr = 0
            asr_p = 0
            flip_asr_p = 0
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            flip_idx_atk_p = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]

            for i, idx in enumerate(idx_atk):
                idx=int(idx)
                sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                ori_node_idx = sub_induct_nodeset[sub_mapping]
                relabeled_node_idx = sub_mapping
                sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)

                # inject trigger on attack test nodes (idx_atk)'''
                with torch.no_grad():
                    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device)
                    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                    # # do pruning in test datas'''
                    induct_edge_index= induct_edge_index[:,induct_edge_weights>0.0].to(device)
                    induct_edge_weights = induct_edge_weights[induct_edge_weights>0.0].to(device)
                    edge_num=sub_induct_edge_index.shape[1]
                    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device,large_graph=False)
                    if args.defense_mode == 'exp':
                        induct_edge_index,induct_edge_weights,xl=prune_with_exp_sage_v4(args,
                                                                    test_model,induct_edge_index,
                                                                    induct_edge_weights,induct_x,
                                                                    device,data.y,sub_mapping,)
                    
                    
                    # attack evaluation

                    output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                    train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
                    output_p = test_model_p(induct_x,induct_edge_index,induct_edge_weights)
                    train_attach_rate_p = (output_p.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
                    ACC+=(output.argmax(dim=1)[relabeled_node_idx]==data.y[idx]).float().mean()
                    # print("Node {}: {}, Origin Label: {}".format(i, idx, data.y[idx]))
                    # print("ASR: {:.4f}".format(train_attach_rate))
                    asr += train_attach_rate
        
                    if(data.y[idx] != args.target_class):
                        flip_asr += train_attach_rate
                    asr_p += train_attach_rate_p
                    if(data.y[idx] != args.target_class):
                        flip_asr_p += train_attach_rate_p
            asr = asr/(idx_atk.shape[0])
            ACC= ACC/(idx_atk.shape[0])
            flip_asr = flip_asr/(flip_idx_atk.shape[0])
            asr_p = asr_p/(idx_atk.shape[0])
            flip_asr_p = flip_asr_p/(flip_idx_atk_p.shape[0])

            print("Overall ASR: {:.4f}".format(asr))
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
            print("Para Overall ASR: {:.4f}".format(asr_p))
            print("Para Flip ASR: {:.4f}/{} nodes".format(flip_asr_p,flip_idx_atk_p.shape[0]))
            print('Remain ACC={:.4f}'.format(ACC))
        elif(args.evaluate_mode == 'overall'):
            # %% inject trigger on attack test nodes (idx_atk)'''
            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
            induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
            # do pruning in test datas'''
            if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
            # attack evaluation

            # test_model = test_model.to(device)
            output = test_model(induct_x,induct_edge_index,induct_edge_weights)
            train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
            print("ASR: {:.4f}".format(train_attach_rate))
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
            ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
            print("CA: {:.4f}".format(ca))
        ACC_seeds.append(float(ACC))
        result_asr.append(float(asr))
        result_acc.append(float(clean_acc))
        result_asr_p.append(float(asr_p))
        result_acc_p.append(float(clean_acc_p))
    print('*'*20)
    print(test_model)
    print('The final ASR:{:.5f}, {:.5f}, Accuracy:{:.5f}, {:.5f}'\
                .format(np.average(result_asr),np.std(result_asr),np.average(result_acc),np.std(result_acc)))
    print('Para The final ASR:{:.5f}, {:.5f}, Accuracy:{:.5f}, {:.5f}'\
                .format(np.average(result_asr_p),np.std(result_asr_p),np.average(result_acc_p),np.std(result_acc_p)))
    print('The final ACC:{:.5f}'\
                .format(np.average(ACC_seeds)))
    print('*'*20)
    models_asr.append(np.average(result_asr))
    models_acc.append(np.average(ACC_seeds))
print('Overall ASR={:.4f}'.format(np.average(models_asr)))
print('Overall ACC={:.4f}'.format(np.average(models_acc)))