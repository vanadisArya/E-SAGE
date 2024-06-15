
# In[1]: 

import argparse
import numpy as np
import torch
from torch_geometric.datasets import Planetoid,Reddit2,Flickr

from ogb.nodeproppred import PygNodePropPredDataset
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated,prune_with_exp,prune_with_exp_sage_v4
from exp_method import Exp_worker


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='Pubmed', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2'])
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=1000, help='Number of epochs to train benign and backdoor model.')
# backdoor setting
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--vs_ratio', type=float, default=0.005,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_size', type=int, default=40,
                    help="ratio of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none','exp'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.1,
                    help="Threshold of prunning edges")
parser.add_argument('--attack_method', type=str, default='Rand_Gene',
                    choices=['Rand_Gene','Rand_Samp','Basic','None'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--trigger_prob', type=float, default=0.5,
                    help="The probability to generate the trigger's edges in random method")
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=3,
                    help="Threshold of prunning edges")
# args = parser.parse_args()
parser.add_argument('--use_target',type=int, default=0,
                    help="use target class to poision")
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
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)
#idx_atk=[i for i in idx_atk if data.y[i]!=args.target_class]
if args.use_target==0:
    idx_atk=idx_atk[data.y[idx_atk]!=args.target_class]
    print('dont use_target')
from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
import random
models = ['GCN','GAT','GraphSage']
# models = ['GCN']
# In[6]: 
models_asr=[]
models_acc=[]
for test_model in models:

    args.test_model = test_model
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000,size=5)
    # seeds = [args.seed]

    result_asr = []
    result_acc = []
    result_asr_c=[]
    result_acc_c=[]
    result_dec=[]
    res_pos=[]
    res_check_rate=[]#多少后加的边被检测了
    for seed in seeds:
        args.seed =seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed.item())
        #torch.cuda.manual_seed(args.seed)
        from heuristic_selection import obtain_attach_nodes
        print(args)
        unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
        #size = args.vs_size #int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
        size=int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
        print('size=',size)
        idx_attach = obtain_attach_nodes(args,unlabeled_idx,size)

        # In[10]:
        # train trigger generator 

        from models.baseline import BaseLine
        model = BaseLine(args,device,data.x)
        # model.fit_rand(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
        poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned_rand(data.x, train_edge_index, data.y,idx_attach)
        # In[12]:
        print("poison_x-data.x=={}".format(poison_x.shape[0]-data.x.shape[0]))
        if(args.defense_mode == 'prune'):
            poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
            bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
        elif(args.defense_mode == 'isolate'):
            poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
            bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
            bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
        else:
            bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
        if(args.attack_method == 'None'):
            bkd_tn_nodes = idx_train
        print("precent of left attach nodes: {:.3f}"\
            .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))
        #%%
        from models.construct import model_construct
        
        test_model = model_construct(args,args.test_model,data,device).to(device) 
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=args.debug)
        output = test_model(poison_x,poison_edge_index,poison_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
        torch.cuda.empty_cache()
        #%%
        induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
        # test_model = test_model.cpu()
        overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
        print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
        # poison_x, poison_edge_index, poison_edge_weights, poison_labels = poison_x.to(device2), poison_edge_index.to(device2), poison_edge_weights.to(device2), poison_labels.to(device2)
        # model.trojan = model.trojan.cpu()
        import time
        time_start = time.time()

        from torch_geometric.utils  import k_hop_subgraph
        idx_atk=idx_atk[:min(len(idx_atk),args.max_atk_num)]

        ASR = 0
        ACC = 0
        exp_idx=[]
        pos_exp=0
        ASR_Clean = []
        ASR_dec=[]
        for idx in idx_atk:
            sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
            relabeled_node_idx = sub_mapping

            # inject trigger on attack test nodes (idx_atk)'''
            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger_rand(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index)
            induct_edge_index= induct_edge_index[:,induct_edge_weights>0.0].to(device)
            induct_edge_weights = induct_edge_weights[induct_edge_weights>0.0].to(device)
            with torch.no_grad():
                if args.defense_mode == 'exp':
                    induct_edge_index,induct_edge_weights,lx=prune_with_exp_sage_v4(args,
                                                                        test_model,induct_edge_index,
                                                                        induct_edge_weights,induct_x,
                                                                        device,data.y,sub_mapping,)
                
                # test_edge_indx.append(induct_edge_index+start)
                # test_x.append(induct_x)
                # test_idx.append(sub_mapping+start)
                
                # start += len(induct_x)
            # test_idx = torch.cat(test_idx)
            # induct_x, induct_edge_index = torch.cat(test_x), torch.cat(test_edge_indx,dim=1)
            # /induct_edge_weights = torch.ones([induct_edge_index.shape[1]],dtype=torch.float32,device=device)
            # # do pruning in test datas'''
                if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                    induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device,large_graph=False)
                # attack evaluation
            #print(induct_edge_index)
            # test_model = test_model.to(device)
                output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                train_attach_rate = (output.argmax(dim=1)[sub_mapping]==args.target_class)
                acc_rate=(output.argmax(dim=1)[sub_mapping]==data.y[idx])
                ASR+=train_attach_rate
                ACC=ACC+acc_rate
        ASR=ASR.item()
        ACC=ACC.item()
        ACC =ACC/len(idx_atk)
        ASR =ASR/len(idx_atk)
        # ASR_dec = torch.cat(ASR_dec).float().mean()
        # ASR_Clean = torch.cat(ASR_Clean).float().mean()
        print("ASR: {:.4f}".format(ASR))
        print("ACC: {:.4f}".format(ACC))
        # print("ASR_Clean: {:.4f}".format(ASR_Clean))
        # print("ASR_dec: {:.4f}".format(ASR_dec))
        result_asr.append(float(ASR))
        result_acc.append(float(ACC))
    print('The final ASR:{:.5f}'.format(np.average(result_asr)))
    print('The final ACC:{:.5f}'.format(np.average(result_acc)))
    models_asr.append(np.average(result_asr))
    models_acc.append(np.average(result_acc))
    # result_dec.append(float(ASR_dec))
    # res_pos.append(float(pos_exp/lenth_exp))
# print('The final ASR:{:.5f}, {:.5f}, Accuracy:{:.5f}, {:.5f},clean_atk ASR:{:.5f},{:.5f},clean_Model_ACC:{:.5f},{:.5f}'\
            # .format(np.average(result_asr),np.std(result_asr),np.average(result_acc),np.std(result_acc),np.average(result_asr_c),np.std(result_asr_c),np.average(result_acc_c),np.std(result_acc_c)))
# %%
# print('The final DEC:{:.5f},{:.5f},pos:{:.5f},{:.5f}'.format(np.average(result_dec),np.std(result_dec),np.average(res_pos),np.std(res_pos)))
# print('The final check rate:{:5f}'.format(np.average(res_check_rate)))
print('Overall ASR={:.4f}'.format(np.average(models_asr)))
print('Overall ACC={:.4f}'.format(np.average(models_acc)))