
#!/usr/bin/env python
# coding: utf-8

# In[1]: 


#import imp
import seaborn as sns
import time
import argparse
import numpy as np
import torch
from exp_method import Exp_worker
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI
import matplotlib.pyplot as plt
import numpy as np

# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated,prune_with_exp,prune_with_exp_sage,prune_with_exp_sage_v2,prune_with_exp_sage_v3,prune_with_exp_sage_v4
import scipy.sparse as sp
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='ogbn-arxiv', 
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
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0.01,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=40,
                    help="number of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="prune",
                    choices=['prune', 'isolate', 'none','exp'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.1,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=100,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.8,
                    help="Threshold of increase similarity")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN','GAT2'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
parser.add_argument('--cut_num', type=int, default=2,
                    help="cut number in exp defense")
parser.add_argument('--use_target_node', type=int, default=0,
                    help="wheather remove target node from idx_atk")
parser.add_argument('--defense_exp_test', type=int, default=0,
                    help="test mode for learn clean acc")
parser.add_argument('--max_atk_num',type=int,default=60,
                    help="size of atk_num")
parser.add_argument('--exp_alg',type=str,default='Captum',
                    choices=['Captum','Attention','GNNExp','GraphMask'],
                    help="explain algorithm")
parser.add_argument('--exp_method',type=str,default='IntegratedGradients',
                    choices=['IntegratedGradients','Saliency','Deconvolution','InputXGradient','GuidedBackprop','ShapleyValueSampling'],
                    help="explain method for captum")
parser.add_argument('--exp_mode',type=str,default='none',choices=['none','Simple'],
                    help="size of atk_num")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
# device=torch.device('cuda:1')
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
if args.use_target_node==0:
    idx_atk=idx_atk[data.y[idx_atk]!=args.target_class]
    print('not use_target')
from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)#test undirected
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]

idx_atk=idx_atk[:args.max_atk_num]

# In[9]:
from torch_geometric.utils  import k_hop_subgraph
from sklearn_extra import cluster
from models.backdoor import Backdoor
from models.construct import model_construct
import heuristic_selection as hs
import sys
# from kmeans_pytorch import kmeans, kmeans_predict

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
# if(args.use_vs_number):
#     size = args.vs_number
# else:
size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
# size = args.vs_number
print("#Attach Nodes:{}".format(size))
# here is randomly select poison nodes from unlabeled nodes
print('SIZE=',size)
if(args.selection_method == 'none'):
    idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
elif(args.selection_method == 'cluster'):
    idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster_degree'):
    if(args.dataset == 'Pubmed'):
        idx_attach = hs.cluster_degree_selection_seperate_fixed(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    else:
        idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
print("idx_attach: {}".format(idx_attach))
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
print(unlabeled_idx)
# In[10]:
# train trigger generator 
model = Backdoor(args,device)
model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
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

models = ['GCN','GAT','GraphSage']

def draw_save_be(data1,data2,data3,name,title,x_lab):
    plt.figure(figsize=(10, 6))
    plt.hist(data1, bins=50, alpha=0.5, label='Clean Model', color='blue')  # 来源1的数据用蓝色表示
    plt.hist(data2, bins=50, alpha=0.5, label='Poisoned Model', color='green')  # 来源2的数据用红色表示
    plt.hist(data3, bins=50, alpha=0.5, label='Poisoned Model+Trigger', color='red')
    plt.xlabel(x_lab,fontsize=18)
    plt.ylabel('Frequency',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title,fontsize=20)
    plt.legend(fontsize=14)
    plt.savefig('/root/dyuan/UGBA/m3_bro/{}'.format(name)+'.pdf',format='pdf')

    print('save to:','/root/dyuan/UGBA/m3_bro/'+name+'.pdf')
def draw_save(data1,data2,data3,name,title,x_lab):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data1, shade=True, label='Clean Model', color='b')  
    sns.kdeplot(data2, shade=True, label='Poisoned Model', color='g')  
    sns.kdeplot(data3,shade=True,  label='Poisoned Model+Trigger', color='r')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(x_lab,fontsize=18)
    plt.ylabel('Frequency',fontsize=18)
    plt.title(title,fontsize=20)
    plt.legend(fontsize=14)
    plt.savefig('/root/dyuan/UGBA/m3_bro/{}'.format(name)+'.pdf',format='pdf')

    print('save to:','/root/dyuan/UGBA/m3_bro/{}'.format(name)+'.pdf')
if args.dataset in ['Flickr','ogbn-arxiv']:small_dataset=False
else:small_dataset=True
for test_model_name in models:

    args.test_model = test_model_name
    # rs = np.random.RandomState(args.seed)
    # seeds = rs.randint(1000,size=5)


    # for seed in seeds:
    #     args.seed = seed
    #     total_time=0
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
    seed=args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print(args)
    #%%
    test_model = model_construct(args,args.test_model,data,device).to(device) 
    test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
    if args.defense_mode == 'exp':
        exp_test=Exp_worker(test_model,data.y,algorithm=args.exp_alg,expl_method=args.exp_method)###

    test_model_p = model_construct(args,args.test_model,data,device).to(device) 
    test_model_p.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)

    #%%
    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])

    exp_test=Exp_worker(test_model,data.y)
    exp_test_p=Exp_worker(test_model_p,data.y)
    overall_induct_edge_index_p, overall_induct_edge_weights_p = train_edge_index.clone(),torch.ones(train_edge_index.shape[1],dtype=torch.float)
    overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
    total_cla=[]
    total_poi=[]
    total_poi_tri=[]
    total_cla_idx=[]
    total_poi_idx=[]
    total_poi_tri_idx=[]
    for i, idx in enumerate(idx_atk):
        idx=int(idx)
        sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
        ori_node_idx = sub_induct_nodeset[sub_mapping]
        relabeled_node_idx = sub_mapping
        sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
        edge_num=sub_induct_edge_index.shape[1]
        degree=0
        for j in sub_induct_edge_index[0]:
            if j==sub_mapping:degree+=1
        with torch.no_grad():
            cla_idx,cla_sc=exp_test_p.get_all(poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_mapping)
            poi_idx,poi_sc=exp_test.get_all(poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_mapping)
            total_cla+=cla_sc[:1]
            total_poi+=poi_sc[:1]
            if len(cla_idx)>0:
                total_cla_idx.append(cla_idx[0]/(sub_induct_edge_index.shape[1]))
            if len(poi_idx)>0:
                total_poi_idx.append(poi_idx[0]/(sub_induct_edge_index.shape[1]))
            # inject trigger on attack test nodes (idx_atk)'''
            # print(cla_sc)
            # print(poi_sc)

            output_c=test_model(poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights)
            acc_c=(output_c.argmax(dim=1)[relabeled_node_idx]==data.y[idx]).float().mean()
            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device)
            
            induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
            #try more
            induct_edge_index= induct_edge_index[:,induct_edge_weights>0.0].to(device)
            induct_edge_weights = induct_edge_weights[induct_edge_weights>0.0].to(device)
            
            poi_tr_idx,poi_tr_sc=exp_test.get_all(induct_x,induct_edge_index,sub_mapping)
            total_poi_tri_idx.append(poi_tr_idx[0]/induct_edge_index.shape[1])
            # print(poi_tr_sc)
            # print('*'*100)
            total_poi_tri+=poi_tr_sc[:1]
            #print('inject_num=',induct_edge_index.shape[1]-sub_induct_edge_index.shape[1])
            # # do pruning in test datas'''
        if (i+1)%40 == 0:print("Now->{:.3f}".format((1+i)/len(idx_atk)))
    array_cla = [t.item() for t in total_cla]
    array_poi = [t.item() for t in total_poi] 
    array_poi_tr = [t.item() for t in total_poi_tri] 
    draw_save(array_cla,array_poi,array_poi_tr,
              'Score_'+test_model_name+'_'+args.dataset,'Distribution of Max Score ('+args.dataset+' & '+test_model_name+')',x_lab='Score')

    draw_save_be(total_cla_idx,total_poi_idx,total_poi_tri_idx,
                 'Pos_'+test_model_name+'_'+args.dataset,'Distribution of Relative Position ('+args.dataset+' & '+test_model_name+')',x_lab='Relative Position')




    test_model = test_model.cpu()
    test_model_p = test_model_p.cpu()
    torch.cuda.empty_cache()
