
#!/usr/bin/env python
# coding: utf-8

# In[1]: 


#import imp
import time
import argparse
import numpy as np
import torch
from exp_method import Exp_worker
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI


# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated,prune_with_exp_sage_v4,sigmoid
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
parser.add_argument('--vs_ratio', type=float, default=0.005,
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
                    choices=['GCN','GAT','GraphSage','GIN','GAT2','GNNGuard'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=3,
                    help="Threshold of prunning edges")
parser.add_argument('--cut_num', type=int, default=2,
                    help="cut number in exp defense")
parser.add_argument('--use_target_node', type=int, default=0,
                    help="wheather remove target node from idx_atk")
parser.add_argument('--defense_exp_test', type=int, default=0,
                    help="test mode for learn clean acc")
parser.add_argument('--max_atk_num',type=int,default=200,
                    help="size of atk_num")
parser.add_argument('--exp_alg',type=str,default='Captum',
                    choices=['Captum','Attention','GNNExp','GraphMask'],
                    help="explain algorithm")
parser.add_argument('--exp_method',type=str,default='IntegratedGradients',
                    choices=['IntegratedGradients','Saliency','Deconvolution','InputXGradient','GuidedBackprop','ShapleyValueSampling'],
                    help="explain method for captum")
parser.add_argument('--exp_mode',type=str,default='none',choices=['none','Simple'],
                    help="size of atk_num")
parser.add_argument('--multi_insert',type=int,default=1,
                    help="number of insert subgraph")
parser.add_argument('--beta',type=float,default=0.5,
                    help="therhold to prune")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
#device=torch.device('cuda:0')
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
if idx_atk.shape[0]>args.max_atk_num:
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
st=time.time()
model = Backdoor(args,device)
model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
ed=time.time()
print(ed-st)
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

models = ['GCN','GAT','GraphSage']
# models=['GraphSage']
total_overall_asr = 0
total_overall_ca = 0
total_overall_asr_p = 0
total_overall_ca_p = 0
total_overall_acc = 0
total_overall_acc_ca=0
if args.dataset in ['Flickr','ogbn-arxiv']:small_dataset=False
else:small_dataset=True
for test_model in models:

    args.test_model = test_model
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000,size=3)
    # seeds = [args.seed]
    overall_asr = 0
    overall_ca = 0
    overall_acc=0
    overall_acc_ca=0
    overall_asr_p = 0
    overall_ca_p = 0
    over_all_time=0
    for seed in seeds:
        dic_pr={1:[],2:[],3:[],4:[]}
        dic_time={1:[],2:[],3:[],4:[]}
        args.seed = seed
        total_time=0
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
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
        output = test_model(poison_x,poison_edge_index,poison_edge_weights)
        output_p = test_model_p(poison_x,poison_edge_index,poison_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
        train_attach_rate_p = (output_p.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
        print("target class rate on Vs_parallel: {:.4f}".format(train_attach_rate_p))
        print("target class rate on Vs: {:.4f}".format(train_attach_rate))
        #%%
        induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
        clean_acc_p = test_model_p.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

        print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
        print("accuracy on Parallel_model clean test nodes: {:.4f}".format(clean_acc_p))
        degree_suc=[]
        degree_all=[]
        if(args.evaluate_mode == '1by1'):
            
            overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
            asr = 0
            flip_asr = 0
            asr_p = 0
            flip_asr_p = 0
            acc = 0
            acc_ca=0
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            for i, idx in enumerate(idx_atk):
                idx=int(idx)
                if i %60==0:print("Now:{:.4f}".format(i/idx_atk.shape[0]))
                sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                ori_node_idx = sub_induct_nodeset[sub_mapping]
                relabeled_node_idx = sub_mapping
                sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                edge_num=sub_induct_edge_index.shape[1]
                degree=0
                for j in sub_induct_edge_index[1]:
                    if j==sub_mapping:degree+=1
                # beta=sigmoid(degree)
                with torch.no_grad():
                    # inject trigger on attack test nodes (idx_atk)'''
                    output_c=test_model(poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights)
                    acc_c=(output_c.argmax(dim=1)[relabeled_node_idx]==data.y[idx]).float().mean()
                    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device)
                    while args.multi_insert>1:
                        induct_x, induct_edge_index,induct_edge_weights=model.inject_trigger(relabeled_node_idx,induct_x,induct_edge_index,induct_edge_weights,device)
                        args.multi_insert-=1
                    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                    #try more
                    induct_edge_index= induct_edge_index[:,induct_edge_weights>0.0].to(device)
                    induct_edge_weights = induct_edge_weights[induct_edge_weights>0.0].to(device)
                    #print('inject_num=',induct_edge_index.shape[1]-sub_induct_edge_index.shape[1])
                    # # do pruning in test datas'''
                    start = time.perf_counter()
                    if args.defense_mode == 'exp':
                        
                        # induct_edge_index,induct_edge_weights=prune_with_exp(args,
                        #                             test_model,induct_edge_index,
                        #                             induct_edge_weights,induct_x,
                        #                             device,data.y,sub_mapping,edge_cla=edge_num)
                        induct_edge_index,induct_edge_weights,exp_meth=prune_with_exp_sage_v4(args,
                                                    test_model,induct_edge_index,
                                                    induct_edge_weights,induct_x,
                                                    device,data.y,sub_mapping,edge_cla=edge_num,
                                                    small_dataset=small_dataset,
                                                    beta=args.beta
                                                    )
                        
                    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device,False)
                    # attack evaluation
                    if args.defense_mode!='exp':exp_meth=1
                    end = time.perf_counter()
                    output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                    ans=output.argmax(dim=1)[relabeled_node_idx]
                    train_attach_rate = (ans==args.target_class).float().mean()
                    acc_def=(ans==data.y[idx]).float().mean()
                    dic_pr[exp_meth].append(train_attach_rate)
                    dic_time[exp_meth].append(end-start)
                    total_time+=(end-start)
                    output_p = test_model_p(induct_x,induct_edge_index,induct_edge_weights)
                    train_attach_rate_p = (output_p.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
                    degree_all.append(degree)
                    if train_attach_rate==1:
                        degree_suc.append(degree)
                    # print('time_avg=',total_time/(1+i))
                asr += train_attach_rate
                acc+=acc_def
                acc_ca+=acc_c
                asr_p += train_attach_rate_p
                if(data.y[idx] != args.target_class):
                    flip_asr += train_attach_rate
                    flip_asr_p += train_attach_rate_p
                
                induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
                output = output.cpu()
                output_p = output_p.cpu()
            total_time=total_time/idx_atk.shape[0]
            acc = acc/(idx_atk.shape[0])
            acc_ca=acc_ca/(idx_atk.shape[0])
            asr = asr/(idx_atk.shape[0])
            asr_p = asr_p/(idx_atk.shape[0])
            flip_asr = flip_asr/(flip_idx_atk.shape[0])
            flip_asr_p = flip_asr_p/(flip_idx_atk.shape[0])
            over_all_time+=total_time
            print('Total time in this atk:{:.4f}'.format(total_time))
            print("Overall ASR: {:.4f},Para Overall ASR:{:.4f}".format(asr,asr_p))
            print("Flip ASR: {:.4f}/{} nodes,Para Flip ASR: {:.4f}/{}".format(flip_asr,flip_idx_atk.shape[0],flip_asr_p,flip_idx_atk.shape[0]))
            print("Overall def_ACC: {:.4f}".format(acc))
            print("Overall Clean_ACC: {:.4f}".format(acc_ca))
            for k,v in dic_pr.items():
                sum=0
                for i in v:sum+=i.item()
                if len(v)>0:
                    ans=sum/len(v)
                    print('Method {},ASR:{:.4f}'.format(k,ans))
            for k,v in dic_time.items():
                sum_time=0
                for i in v:sum_time+=i
                if len(v)>0:
                    ans_time=sum_time/len(v)
                    print('Method {},AVG_TIME:{:.4f}'.format(k,ans_time))
        elif(args.evaluate_mode == 'overall'):
            # %% inject trigger on attack test nodes (idx_atk)'''
            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
            induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
            # do pruning in test datas'''
            if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
            # attack evaluation
            output = test_model(induct_x,induct_edge_index,induct_edge_weights)
            train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
            print("ASR: {:.4f}".format(train_attach_rate))
            asr = train_attach_rate
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
            ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
            print("CA: {:.4f}".format(ca))

            induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
            output = output.cpu()
        print('degree of atk success node',degree_suc)
        print('degree of all akt nodes',degree_all)
        overall_asr += asr
        overall_acc += acc
        overall_acc_ca+=acc_ca
        overall_ca += clean_acc
        overall_asr_p += asr_p
        overall_ca_p += clean_acc_p
        test_model = test_model.cpu()
        test_model_p = test_model_p.cpu()
        torch.cuda.empty_cache()
    overall_asr = overall_asr/len(seeds)
    over_all_time=over_all_time/len(seeds)
    overall_ca = overall_ca/len(seeds)
    overall_asr_p = overall_asr_p/len(seeds)
    overall_ca_p = overall_ca_p/len(seeds)
    overall_acc = overall_acc/len(seeds)
    overall_acc_ca = overall_acc_ca/len(seeds)
    print("Overall def_acc: {:.4f}, Seed: {})".format(overall_acc, args.seed))
    print("Overall clean_acc: {:.4f}, Seed: {})".format(overall_acc_ca, args.seed))
    print("Overall ASR: {:.4f},Para Overall ASR: {:.4f} ({} model, Seed: {})".format(overall_asr, overall_asr_p,args.test_model, args.seed))
    print("Overall Clean Accuracy: {:.4f},Para Overall Clean ACC:{:.4f}".format(overall_ca,overall_ca_p))
    print("Overall Time: {:.4f}, Seed: {})".format(over_all_time, args.seed))
    total_overall_asr += overall_asr
    total_overall_ca += overall_ca
    total_overall_asr_p += overall_asr_p
    total_overall_ca_p += overall_ca_p
    total_overall_acc+=overall_acc
    total_overall_acc_ca+=overall_acc_ca
    test_model.to(torch.device('cpu'))
    test_model_p.to(torch.device('cpu'))
    torch.cuda.empty_cache()
total_overall_asr = total_overall_asr/len(models)
total_overall_ca = total_overall_ca/len(models)
total_overall_asr_p = total_overall_asr_p/len(models)
total_overall_ca_p = total_overall_ca_p/len(models)
total_overall_acc=total_overall_acc/len(models)
print("Total Overall ASR: {:.4f} ".format(total_overall_asr))
print("Total Clean Accuracy: {:.4f}".format(total_overall_ca))
print("Para Total Overall ASR: {:.4f} ".format(total_overall_asr_p))
print("Para Total Clean Accuracy: {:.4f}".format(total_overall_ca_p))
print("Total def Clean Accuracy: {:.4f}".format(total_overall_acc))