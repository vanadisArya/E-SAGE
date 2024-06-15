import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj,dense_to_sparse
import torch
from torch_geometric.utils import to_undirected
import scipy.sparse as sp
from torch_geometric.utils  import k_hop_subgraph
from exp_method import Exp_worker
def sigmoid(x, k=1.2, x0=3.5):
    x=min(x,10)
    return 0.35 / (1 + np.exp(k * (x - x0))) + 0.45
def edge_sim_analysis(edge_index, features):
    sims = []
    for (u,v) in edge_index:
        sims.append(float(F.cosine_similarity(features[u].unsqueeze(0),features[v].unsqueeze(0))))
    sims = np.array(sims)
    # print(f"mean: {sims.mean()}, <0.1: {sum(sims<0.1)}/{sims.shape[0]}")
    return sims

def prune_unrelated_edge(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    # update structure
    
    updated_edge_index = edge_index[:,edge_sims>args.prune_thr]
    updated_edge_weights = edge_weights[edge_sims>args.prune_thr]
    #print('edge_weights==',edge_weights.shape[0])
    #print('updated_edge_weights==',updated_edge_weights.shape[0])
    # if updated_edge_index.shape[1]!=edge_index.shape[1]:
    #     print('prune edge_num=',edge_index.shape[1]-updated_edge_index.shape[1])
    return updated_edge_index,updated_edge_weights

def prune_unrelated_edge_isolated(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        # calculate edge simlarity
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    dissim_edges_index = np.where(edge_sims.cpu()<=args.prune_thr)[0]
    if dissim_edges_index.shape!=0:
        print(dissim_edges_index.shape[0])
    edge_weights[dissim_edges_index] = 0
    # select the nodes between dissimilar edgesy
    dissim_edges = edge_index[:,dissim_edges_index]    # output: [[v_1,v_2],[u_1,u_2]]
    dissim_nodes = torch.cat([dissim_edges[0],dissim_edges[1]]).tolist()
    dissim_nodes = list(set(dissim_nodes))
    # update structure
    updated_edge_index = edge_index[:,edge_weights>0.0]
    updated_edge_weights = edge_weights[edge_weights>0.0]
    #print('Pruned edges number:{}'.format(edge_index.shape[0]-updated_edge_index.shape[0]))
    return updated_edge_index,updated_edge_weights,dissim_nodes 

def select_target_nodes(args,seed,model,features,edge_index,edge_weights,labels,idx_val,idx_test):
    test_ca,test_correct_index = model.test_with_correct_nodes(features,edge_index,edge_weights,labels,idx_test)
    test_correct_index = test_correct_index.tolist()
    '''select target test nodes'''
    test_correct_nodes = idx_test[test_correct_index].tolist()
    # filter out the test nodes that are not in target class
    target_class_nodes_test = [int(nid) for nid in idx_test
            if labels[nid]==args.target_class] 
    # get the target test nodes
    idx_val,idx_test = idx_val.tolist(),idx_test.tolist()
    rs = np.random.RandomState(seed)
    cand_atk_test_nodes = list(set(test_correct_nodes) - set(target_class_nodes_test))  # the test nodes not in target class is candidate atk_test_nodes
    atk_test_nodes = rs.choice(cand_atk_test_nodes, args.target_test_nodes_num)
    '''select clean test nodes'''
    cand_clean_test_nodes = list(set(idx_test) - set(atk_test_nodes))
    clean_test_nodes = rs.choice(cand_clean_test_nodes, args.clean_test_nodes_num)
    '''select poisoning nodes from unlabeled nodes (assign labels is easier than change, also we can try to select from labeled nodes)'''
    N = features.shape[0]
    cand_poi_train_nodes = list(set(idx_val)-set(atk_test_nodes)-set(clean_test_nodes))
    poison_nodes_num = int(N * args.vs_ratio)
    poi_train_nodes = rs.choice(cand_poi_train_nodes, poison_nodes_num)
    
    return atk_test_nodes, clean_test_nodes,poi_train_nodes

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()


def prune_with_exp(args,model,edge_index,edge_weights,x,
                   device,y,target,beta=1.25,edge_cla=100000,debug=1):
    if edge_index[0].shape[0]==0:return edge_index,edge_weights
    exp_test=Exp_worker(model,y,algorithm=args.exp_alg)
    #exp within a radius of 1
    if edge_weights.shape[0]>50:args.exp_mode='Simple'
    else:args.exp_mode='none'
    if args.exp_mode=='Simple':
        if edge_index[0].shape[0]==0:return edge_index,edge_weights
        sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [target], num_hops = 1, edge_index = edge_index, relabel_nodes=True)
        relabeled_node_idx = sub_mapping
        ori_edge_map=[]
        for i,edge in enumerate(sub_edge_mask):
            if edge:ori_edge_map.append(i)
        imp_edge_index,imp_edge_score,dict_feat=exp_test.get_all(x[sub_induct_nodeset],sub_induct_edge_index,sub_mapping)
        dim_target=0
        for i in sub_induct_edge_index[0]:
            if i==sub_mapping:dim_target+=1
        if dim_target==0:return edge_index,edge_weights
        max_cut=max(int(dim_target/3),2)
        cut=0
        if len(imp_edge_score)>0:
            max_imp=imp_edge_score[0]
        for idx_1,edge in enumerate(imp_edge_index):
            edge_weights[ori_edge_map[edge]]=0
            if debug==1:
                print(ori_edge_map[edge],' ','cla_edge=',edge_cla)
                print('*'*20)
            break
            if idx_1<len(imp_edge_score)-1 and idx_1<1:
                if imp_edge_score[idx_1]>imp_edge_score[idx_1+1]*beta:
                    edge_weights[ori_edge_map[edge]]=0
                    #print(imp_edge_score)
                    break
            #if imp_edge_score[idx_1]<beta or imp_edge_score[idx_1]<max_imp/2:break

            #if cut>=max_cut:break
        new_edge_index=edge_index[:,edge_weights>0.0].to(device)
        new_edge_weights=edge_weights[edge_weights>0.0].to(device)
        return new_edge_index,new_edge_weights
    else:
        imp_edge_index,imp_edge_score,dict_feat=exp_test.get_all(x, edge_index,target)
        cut=0
        if debug==1:
            print('cla_edge=',edge_cla)
            print(imp_edge_index)
            l=[]
            for i in imp_edge_score:
                l.append(torch.round(i, decimals=4).item())
            print(l)
            print(edge_index)
            print('target:')
            g_0=[]
            g_1=[]
            for i in range(max(3,len(imp_edge_index))):
                g_0.append(edge_index[0][imp_edge_index[i]].item())
                g_1.append(edge_index[1][imp_edge_index[i]].item())
            print(g_0)
            print(g_1)
            
            print('--'*10)
        if len(imp_edge_score)>0:
            max_imp=imp_edge_score[0]
        dim_target=0
        for i in edge_index[0]:
            if i==target:dim_target+=1
        max_cut=max(int(dim_target  /3),2)
        for idx_1,edge in enumerate(imp_edge_index):
            edge_weights[edge]=0
            cut+=1
            if idx_1<len(imp_edge_score)-1:
                if imp_edge_score[idx_1+1]<max_imp/beta:
                    break
            if cut>=max_cut:break
        updated_edge_index = edge_index[:,edge_weights>0.0].to(device)
        updated_edge_weights = edge_weights[edge_weights>0.0].to(device)
    return updated_edge_index,updated_edge_weights

def debug_sage(edge_index,imp_score,imp_idx,edge_index_now,edge_cla):
    print('edge_cla=',edge_cla)
    print('before_edge_index=',edge_index)
    print('edge_index_now=',edge_index_now)
    print('imp_score=',imp_score)
    print('imp_idx',imp_idx)
def prune_with_exp_sage(args,model,edge_index,edge_weights,x,
                   device,y,target,debug=1,node_bound=3,edge_bound=50,edge_cla=10000):

    if edge_index[0].shape[0]==0:return edge_index,edge_weights
    edge_index_before=edge_index.clone()
    exp_test=Exp_worker(model,y)
    sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [target], num_hops = 1, edge_index = edge_index, relabel_nodes=True)
    if edge_weights.shape[0]<edge_bound:
        # print('simple')
        imp_edge_index,imp_edge_score,dict_feat=exp_test.get_all(x, edge_index,target)
        if len(imp_edge_score)>0:
            edge_weights[imp_edge_index[0]]=0
        updated_edge_index = edge_index[:,edge_weights>0.0].to(device)
        updated_edge_weights = edge_weights[edge_weights>0.0].to(device)
        return updated_edge_index,updated_edge_weights,1
    # elif x.shape[0]>1000:
    #     # print('r=1')
    #     if edge_index[0].shape[0]==0:return edge_index,edge_weights
    #     ori_edge_map=[]
    #     for i,edge in enumerate(sub_edge_mask):
    #         if edge:ori_edge_map.append(i)
    #     imp_edge_index,imp_edge_score,dict_feat=exp_test.get_all(x[sub_induct_nodeset],sub_induct_edge_index,sub_mapping)
    #     dim_target=0
    #     for i in sub_induct_edge_index[0]:
    #         if i==sub_mapping:dim_target+=1
    #     if dim_target==0:return edge_index,edge_weights
    #     if len(imp_edge_score)>0:
    #         edge_weights[ori_edge_map[edge]]=0
    #     new_edge_index=edge_index[:,edge_weights>0.0].to(device)
    #     new_edge_weights=edge_weights[edge_weights>0.0].to(device)
    #     return new_edge_index,new_edge_weights,2
    else:
        # print('sage')
        # print('edge_weights=',edge_weights)
        while node_bound>1 and (len(sub_induct_nodeset)-1)*(1+node_bound)>edge_bound:
            node_bound-=1
        dict_cnt_one_hop={}
        # print('node_bound:',node_bound)
        for i in sub_induct_nodeset:
            if i!=target:dict_cnt_one_hop[i.item()]=0
            else:dict_cnt_one_hop[i.item()]=-10000
        # print('dict_cnt_one_hop:',dict_cnt_one_hop)
        edge_weights_new=torch.zeros(edge_weights.shape[0],dtype=torch.float)
        # print('edge_weights-shape',edge_weights.shape)
        # print('edge_weights',edge_weights)
        # print('edge_weights_new-shape',edge_weights_new.shape)
        # print('edge_weights_new',edge_weights_new)
        for idx,edge in enumerate(edge_index[1]):
            edge=edge.item()
            if edge in dict_cnt_one_hop:
                if dict_cnt_one_hop[edge]<node_bound:
                    dict_cnt_one_hop[edge]+=1
                    edge_weights_new[idx]=1
            if edge_index[0][idx].item() in dict_cnt_one_hop:edge_weights_new[idx]=1
        edge_index_new=edge_index[:,edge_weights_new>0.0]
        edge_map=[]
        
        for i in range(edge_index.shape[1]):#map new_edge_idx ->edge_idx
            if edge_weights_new[i]>0.0:edge_map.append(i)
        imp_edge_index,imp_edge_score,dict_feat=exp_test.get_all(x, edge_index_new,target)
        if debug==1:
            print('target=',target)
            print('after_sage',edge_index_new)
            print('edge_map',edge_map)
            print('imp_edge_index[0]',imp_edge_index[0])
            print('edge_map[imp_edge_index[0]',edge_map[imp_edge_index[0]])
        if len(imp_edge_score)>0:
            edge_weights[edge_map[imp_edge_index[0]]]=0

        updated_edge_index = edge_index[:,edge_weights>0.0].to(device)
        updated_edge_weights = edge_weights[edge_weights>0.0].to(device)
    #exp within a radius of 1
        if debug==1:
            debug_sage(edge_index_before,imp_edge_score,imp_edge_index,updated_edge_index,edge_cla)
        return updated_edge_index,updated_edge_weights,3
def prune_with_exp_sage_v2(args,model,edge_index,edge_weights,x,
                   device,y,target,debug=1,node_bound=3,edge_bound=100,edge_cla=10000,small_dataset=True):

    if edge_index[0].shape[0]==0:return edge_index,edge_weights
    edge_index_before=edge_index.clone()
    exp_test=Exp_worker(model,y)
    sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [target], num_hops = 1, edge_index = edge_index, relabel_nodes=True)
    if small_dataset or edge_weights.shape[0]<edge_bound:
        # print('simple')
        imp_edge_index,imp_edge_score=exp_test.get_all(x, edge_index,target)
        if len(imp_edge_score)>0:
            edge_weights[imp_edge_index[0]]=0
            if debug==1:
                print('cla_edge=',edge_cla)
                print(imp_edge_index)
                l=[]
                for i in imp_edge_score:
                    l.append(torch.round(i, decimals=4).item())
                print(l)
                print(edge_index)
                print('target:')
                g_0=[]
                g_1=[]
                for i in range(max(3,len(imp_edge_index))):
                    g_0.append(edge_index[0][imp_edge_index[i]].item())
                    g_1.append(edge_index[1][imp_edge_index[i]].item())
                    if i>=len(imp_edge_score)-1:break
                print(g_0)
                print(g_1)
                
                print('--'*10)
        updated_edge_index = edge_index[:,edge_weights>0.0].to(device)
        updated_edge_weights = edge_weights[edge_weights>0.0].to(device)
        return updated_edge_index,updated_edge_weights,1

    else:
        # print('sage')
        # print('edge_weights=',edge_weights)
        # while node_bound>1 and (len(sub_induct_nodeset)-1)*(1+node_bound)>edge_bound:
        #     node_bound-=1
        # if len(sub_induct_nodeset)>15:node_bound-=1
        dict_cnt_one_hop={}

        # print('node_bound:',node_bound)
        for i in sub_induct_nodeset:
            if i!=target:dict_cnt_one_hop[i.item()]=0
            else:dict_cnt_one_hop[i.item()]=-10000
        # print('dict_cnt_one_hop:',dict_cnt_one_hop)
        edge_weights_new=torch.zeros(edge_weights.shape[0],dtype=torch.float)
        # print('edge_weights-shape',edge_weights.shape)
        # print('edge_weights',edge_weights)
        # print('edge_weights_new-shape',edge_weights_new.shape)
        # print('edge_weights_new',edge_weights_new)
        node_subset=[]
        for idx,edge in enumerate(edge_index[1]):
            edge=edge.item()
            if edge in dict_cnt_one_hop:
                if dict_cnt_one_hop[edge]<node_bound:
                    dict_cnt_one_hop[edge]+=1
                    edge_weights_new[idx]=1
                    node_subset.append(edge_index[0][idx].item())
                    node_subset.append(edge_index[1][idx].item())
            # if edge_index[0][idx].item() in dict_cnt_one_hop:
            #     edge_weights_new[idx]=1
        # print('node_subset',node_subset)
        # print('sub_induct_nodeset',sub_induct_nodeset)
        total_subset_node=list(set(node_subset+sub_induct_nodeset.tolist()))
        # print('total_subset_node',total_subset_node)
        total_subset_node=sorted(total_subset_node)
        # print('total_subset_node',total_subset_node)
        l_temp=list(range(0,len(total_subset_node)))
        node_map_to_new=dict(zip(total_subset_node,l_temp))
        # print('node_map_to_new',node_map_to_new)
        x_new=x[total_subset_node].to(device)
        edge_index_new=edge_index[:,edge_weights_new>0.0]
        # edge_index_new=to_undirected(edge_index_new)
        edge_map=[]
        # print('edge_index_new',edge_index_new)
        for idx in range(len(edge_index_new[0])):
            edge_index_new[0][idx]=node_map_to_new[edge_index_new[0][idx].item()]
            edge_index_new[1][idx]=node_map_to_new[edge_index_new[1][idx].item()]
        # print(edge_index_new)
        target_now=node_map_to_new[target.item()]
        for i in range(edge_index.shape[1]):#map new_edge_idx ->edge_idx
            if edge_weights_new[i]>0.0:edge_map.append(i)
        imp_edge_index,imp_edge_score=exp_test.get_all(x_new, edge_index_new,target_now)
        if debug==1:
            print('target=',target)
            print('after_sage',edge_index_new)
            print('edge_map',edge_map)
            print('imp_edge_index[0]',imp_edge_index[0])
            print('edge_map[imp_edge_index[0]',edge_map[imp_edge_index[0]])
        if len(imp_edge_score)>0:
            edge_weights[edge_map[imp_edge_index[0]]]=0

        updated_edge_index = edge_index[:,edge_weights>0.0].to(device)
        updated_edge_weights = edge_weights[edge_weights>0.0].to(device)
    #exp within a radius of 1
        if debug==1:
            debug_sage(edge_index_before,imp_edge_score,imp_edge_index,updated_edge_index,edge_cla)
        return updated_edge_index,updated_edge_weights,3
def prune_with_exp_sage_v3(args,model,edge_index,edge_weights,x,
                   device,y,target,beta=0.5,debug=0,node_bound=3,edge_bound=1000,edge_cla=10000,small_dataset=True,atk_mode=None):

    if edge_index[0].shape[0]==0:return edge_index,edge_weights
    edge_index_before=edge_index.clone()
    exp_test=Exp_worker(model,y)
    cut=0
    sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [target], num_hops = 1, edge_index = edge_index, relabel_nodes=True)
    if small_dataset or edge_weights.shape[0]<edge_bound:
        # print('simple')
        imp_edge_index,imp_edge_score=exp_test.get_all(x, edge_index,target)
        lenth=len(imp_edge_index)


        while(lenth>0):
            fo=['{:.2f}'.format(j.item()) for j in imp_edge_score]
            print(fo)
            if len(imp_edge_score)>0:
            
                if debug==1:
                    print('cla_edge=',edge_cla)
                    print(imp_edge_index)
                    l=[]
                    for i in imp_edge_score:
                        l.append(torch.round(i, decimals=4).item())
                    print(l)
                    print(edge_index)
                    print('target:')
                    g_0=[]
                    g_1=[]
                    for i in range(max(3,len(imp_edge_index))):
                        g_0.append(edge_index[0][imp_edge_index[i]].item())
                        g_1.append(edge_index[1][imp_edge_index[i]].item())
                        if i>=len(imp_edge_score)-1:break
                    print(g_0)
                    print(g_1)
                    
                    print('--'*10)
            if imp_edge_score[0]<beta:break
            for i,edge in enumerate(imp_edge_score):
                if edge>beta:
                    edge_weights[imp_edge_index[i]]=0
                    cut+=1
            edge_index = edge_index[:,edge_weights>0.0].to(device)
            edge_weights = edge_weights[edge_weights>0.0].to(device)
            imp_edge_index,imp_edge_score=exp_test.get_all(x, edge_index,target)
            lenth=len(imp_edge_index)
        print('*'*20)
        return edge_index,edge_weights,1

    else:

        dict_cnt_one_hop={}

        # print('node_bound:',node_bound)
        for i in sub_induct_nodeset:
            if i!=target:dict_cnt_one_hop[i.item()]=0
            else:dict_cnt_one_hop[i.item()]=-10000
        # print('dict_cnt_one_hop:',dict_cnt_one_hop)
        edge_weights_new=torch.zeros(edge_weights.shape[0],dtype=torch.float)
        # print('edge_weights-shape',edge_weights.shape)
        # print('edge_weights',edge_weights)
        # print('edge_weights_new-shape',edge_weights_new.shape)
        # print('edge_weights_new',edge_weights_new)
        node_subset=[]
        for idx,edge in enumerate(edge_index[1]):
            edge=edge.item()
            if edge in dict_cnt_one_hop:
                if dict_cnt_one_hop[edge]<node_bound:
                    dict_cnt_one_hop[edge]+=1
                    edge_weights_new[idx]=1
                    node_subset.append(edge_index[0][idx].item())
                    node_subset.append(edge_index[1][idx].item())

        total_subset_node=list(set(node_subset+sub_induct_nodeset.tolist()))
        # print('total_subset_node',total_subset_node)
        total_subset_node=sorted(total_subset_node)
        # print('total_subset_node',total_subset_node)
        l_temp=list(range(0,len(total_subset_node)))
        node_map_to_new=dict(zip(total_subset_node,l_temp))
        # print('node_map_to_new',node_map_to_new)
        x_new=x[total_subset_node].to(device)
        edge_index_new=edge_index[:,edge_weights_new>0.0]
        # edge_index_new=to_undirected(edge_index_new)
        edge_map=[]
        # print('edge_index_new',edge_index_new)
        for idx in range(len(edge_index_new[0])):
            edge_index_new[0][idx]=node_map_to_new[edge_index_new[0][idx].item()]
            edge_index_new[1][idx]=node_map_to_new[edge_index_new[1][idx].item()]
        # print(edge_index_new)
        target_now=node_map_to_new[target.item()]
        for i in range(edge_index.shape[1]):#map new_edge_idx ->edge_idx
            if edge_weights_new[i]>0.0:edge_map.append(i)
        imp_edge_index,imp_edge_score=exp_test.get_all(x_new, edge_index_new,target_now)
        if debug==1:
            print('target=',target)
            print('after_sage',edge_index_new)
            print('edge_map',edge_map)
            print('imp_edge_index[0]',imp_edge_index[0])
            print('edge_map[imp_edge_index[0]',edge_map[imp_edge_index[0]])
        if len(imp_edge_score)>0:
            edge_weights[edge_map[imp_edge_index[0]]]=0

        updated_edge_index = edge_index[:,edge_weights>0.0].to(device)
        updated_edge_weights = edge_weights[edge_weights>0.0].to(device)
    #exp within a radius of 1
        if debug==1:
            debug_sage(edge_index_before,imp_edge_score,imp_edge_index,updated_edge_index,edge_cla)
        return updated_edge_index,updated_edge_weights,3
def prune_with_exp_sage_v4(args,model,edge_index,edge_weights,x,
                   device,y,target,alpha=5,beta=0.4,debug=0,node_bound=5,edge_bound=300,edge_cla=10000,small_dataset=False,atk_mode=None):

    if edge_index[0].shape[0]==0:
        return edge_index,edge_weights
    edge_index_before=edge_index.clone()
    exp_test=Exp_worker(model,y)
    cut=0
    degree=0
    for i in edge_index[1]:
        if i ==target:degree+=1
    beta=sigmoid(degree)
    max_cut=max(int(degree/4),4)
    if atk_mode=='TDGIA':
        max_cut=100
    # print('max_cut=',max_cut)
    # print('beta=',beta)
    # print('max_cut=',max_cut)
    sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [target], num_hops = 1, edge_index = edge_index, relabel_nodes=True)
    if edge_weights.shape[0]<edge_bound:

        
        # print('simple')
        imp_edge_index,imp_edge_score=exp_test.get_all(x, edge_index,target)
        # print(imp_edge_score)
        lenth=len(imp_edge_index)
        while(lenth>0):
            # print(imp_edge_score)
            is_Cliff=False
            if lenth>1:
                if imp_edge_score[0]>imp_edge_score[1]*alpha:is_Cliff=True
            if imp_edge_score[0]<beta and not is_Cliff:
                # print('break')
                break
            edge_weights[imp_edge_index[0]]=0
            # print(edge_index[0][imp_edge_index[0]].item(),'*',edge_index[1][imp_edge_index[0]].item())
            cut+=1
            if cut>=max_cut:break
            edge_index = edge_index[:,edge_weights>0.0].to(device)
            edge_weights = edge_weights[edge_weights>0.0].to(device)
            imp_edge_index,imp_edge_score=exp_test.get_all(x, edge_index,target)
            lenth=len(imp_edge_index)
            # print(imp_edge_score)

        # print('#'*20)
        return edge_index,edge_weights,1
    else:

        dict_cnt_one_hop={}
        # print('edge_weights.shape=',edge_weights.shape[0])
        # print('node_bound:',node_bound)
        for i in sub_induct_nodeset:
            if i!=target:dict_cnt_one_hop[i.item()]=0
            else:dict_cnt_one_hop[i.item()]=-1000000
        # print('dict_cnt_one_hop:',dict_cnt_one_hop)
        edge_weights_new=torch.zeros(edge_weights.shape[0],dtype=torch.float)
        # print('edge_weights-shape',edge_weights.shape)
        # print('edge_weights',edge_weights)
        # print('edge_weights_new-shape',edge_weights_new.shape)
        # print('edge_weights_new',edge_weights_new)
        node_subset=[]
        for idx,edge in enumerate(edge_index[1]):
            edge=edge.item()
            if edge in dict_cnt_one_hop:
                if dict_cnt_one_hop[edge]<node_bound or edge_index[0][idx]==target:
                    if edge_index[0][idx]!=target:dict_cnt_one_hop[edge]+=1
                    edge_weights_new[idx]=1
                    node_subset.append(edge_index[0][idx].item())
                    node_subset.append(edge_index[1][idx].item())

        total_subset_node=list(set(node_subset+sub_induct_nodeset.tolist()))
        # print('total_subset_node',total_subset_node)
        total_subset_node=sorted(total_subset_node)
        # if len(sub_induct_nodeset)>15:
        #     print('sub_induct_nodeset',sub_induct_nodeset)
        #     print('len(sub_induct_nodeset)',sub_induct_nodeset)
        #     print('total_subset_node',total_subset_node)
        #     print('len(total_subset_node)=',len(total_subset_node))
        # print('total_subset_node',total_subset_node)
        l_temp=list(range(0,len(total_subset_node)))
        node_map_to_new=dict(zip(total_subset_node,l_temp))
        node_map_to_old=dict(zip(l_temp,total_subset_node))
        # print('node_map_to_new',node_map_to_new)
        x_new=x[total_subset_node].to(device)
        edge_index_new=edge_index[:,edge_weights_new>0.0]
        # edge_index_new=to_undirected(edge_index_new)
        edge_map=[]
        # print('edge_index_new',edge_index_new)
        for idx in range(len(edge_index_new[0])):
            edge_index_new[0][idx]=node_map_to_new[edge_index_new[0][idx].item()]
            edge_index_new[1][idx]=node_map_to_new[edge_index_new[1][idx].item()]
        # print(edge_index_new)
        target_now=node_map_to_new[target.item()]
        for i in range(edge_index.shape[1]):#map new_edge_idx ->edge_idx
            if edge_weights_new[i]>0.0:edge_map.append(i)
        
        imp_edge_index,imp_edge_score=exp_test.get_all(x_new, edge_index_new,target_now)
        # print(imp_edge_score)
        if debug==1:
            print('target=',target)
            print('after_sage',edge_index_new)
            print('edge_map',edge_map)
            print('imp_edge_index[0]',imp_edge_index[0])
            print('edge_map[imp_edge_index[0]',edge_map[imp_edge_index[0]])
        edge_weights_use=torch.ones(edge_index_new.shape[1],dtype=torch.float)
#
        rec_del_idx=[]
        rec_del_idx_same=[]
        same_p=0
        lenth=len(imp_edge_index)
        while(lenth>0):
            # print(imp_edge_score)
            is_Cliff=False
            if lenth>1:
                if imp_edge_score[0]>imp_edge_score[1]*alpha:is_Cliff=True
            if imp_edge_score[0]<beta and not is_Cliff:break
            # if imp_edge_score[0]>beta:
            #     for i,edge in enumerate(imp_edge_score):
            #         if edge>beta:
            edge_weights_use[imp_edge_index[0]]=0
            rec_del_idx.append(imp_edge_index[0])
            rec_del_idx_same.append(same_p)
            cut+=1
            # else:
            #     cut+=1
            #     rec_del_idx.append(imp_edge_index[0])
            #     edge_weights_use[imp_edge_index[0]]=0
            #     rec_del_idx_same.append(same_p)
            if cut>=max_cut:break
            edge_index_new = edge_index_new[:,edge_weights_use>0.0].to(device)
            edge_weights_use = edge_weights_use[edge_weights_use>0.0].to(device)
            imp_edge_index,imp_edge_score=exp_test.get_all(x_new, edge_index_new,target_now)
            lenth=len(imp_edge_index)
            # print(imp_edge_score)
            same_p+=1
            # print('edge_weights_use.shape=',edge_weights_use.shape[0])
        # fix_idx=[]
        # print('rec_del_idx',rec_del_idx)
        # print('***')
        for i,edge in enumerate(rec_del_idx):
            fix_idx_i=0
            for j in range(i):
                if edge>=rec_del_idx[j] and rec_del_idx_same[j]<rec_del_idx_same[i]:fix_idx_i+=1
            # fix_idx.append(edge+fix_idx_i)
            # print('edge+fix_idx_i=',edge+fix_idx_i)
            # print('edge_map[edge+fix_idx_i]=',edge_map[edge+fix_idx_i])
            edge_weights[edge_map[edge+fix_idx_i]]=0
        updated_edge_index = edge_index[:,edge_weights>0.0].to(device)
        updated_edge_weights = edge_weights[edge_weights>0.0].to(device)
    #exp within a radius of 1
        if debug==1:
            debug_sage(edge_index_before,imp_edge_score,imp_edge_index,updated_edge_index,edge_cla)
        return updated_edge_index,updated_edge_weights,3