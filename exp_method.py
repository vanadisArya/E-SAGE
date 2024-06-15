import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer,AttentionExplainer,GraphMaskExplainer
import numpy as np
class Exp_worker():
    def __init__(self,
                 model,
                 y,
                 algorithm='Captum',
                 node_mask_type='attributes',
                 edge_mask_type='object',
                 task_level='node',
                 return_type='log_probs',
                 expl_method='IntegratedGradients'
                 ):
        classes=y.max()+1
        if classes>2:self.mode='multiclass_classification'
        elif classes==2:self.mode='binary_classification'
        else:raise TypeError('Illegal data set')
        if algorithm=='GNNExp':
            self.Exper=Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=200),
                explanation_type='model',
                node_mask_type= node_mask_type,
                edge_mask_type=edge_mask_type,
                model_config=dict(
                    mode=self.mode,
                    task_level=task_level,
                    return_type=return_type,
                ),
            )
        elif algorithm=='Captum':
            self.Exper=Explainer(
                model=model,
                algorithm=CaptumExplainer(expl_method),
                explanation_type='model',
                node_mask_type= node_mask_type,
                edge_mask_type='object',
                model_config=dict(
                    mode=self.mode,
                    task_level=task_level,
                    return_type=return_type,
                ),
            )
        elif algorithm=='Attention':
            self.Exper=Explainer(
                model=model,
                algorithm=AttentionExplainer(),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode=self.mode,
                    task_level=task_level,
                    return_type=return_type,
                ),
            )
        elif algorithm=='GraphMask':
            self.Exper=Explainer(
                model=model,
                algorithm=GraphMaskExplainer(2, epochs=5),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode=self.mode,
                    task_level=task_level,
                    return_type=return_type,
                ),
            )
    
    def get_expl(self,x,edge_index, node_index):
        return self.Exper(x,edge_index,index=node_index)
    
    
    def get_important_edges(self,x,edge_index, node_index,top=10):
        explanation=self.Exper(x,edge_index,index=node_index)
        edge_mask=explanation.get('edge_mask')
        ans=[]
        for idx,score in enumerate(edge_mask):
            if score>0.0:
                ans.append((score,idx))
        ans=sorted(ans,key=lambda x: x[0],reverse=True)
        imp_edge_index=[i[1] for i in ans[0:top]]
        imp_edge_score=[i[0] for i in ans[0:top]]
        #return torch.Tensor(imp_edge_index),torch.Tensor(imp_edge_score)
        #return dict(zip(imp_edge_index,imp_edge_score))
        return [imp_edge_index,imp_edge_score]
    def get_important_features(self,x,edge_index, node_index,top=10):
        explanation=self.Exper(x,edge_index,index=node_index)
        dict_feat=explanation.visualize_feature_importance(path=None, top_k=top,visualize=False).to_dict()['score']
        return dict_feat
    def get_all(self,x,edge_index, node_index,top=10):
        explanation=self.Exper(x,edge_index,index=node_index)
        edge_mask=explanation.get('edge_mask')
        ans=[]
        for idx,score in enumerate(edge_mask):
            if score>0.0:
                ans.append((score,idx))
        ans=sorted(ans,key=lambda x: x[0],reverse=True)
        imp_edge_index=[i[1] for i in ans[0:top]]
        imp_edge_score=[i[0] for i in ans[0:top]]
        #return torch.Tensor(imp_edge_index),torch.Tensor(imp_edge_score)
        #return dict(zip(imp_edge_index,imp_edge_score))
        explanation=self.Exper(x,edge_index,index=node_index)
        # dict_feat=explanation.visualize_feature_importance(path=None, top_k=top,visualize=False).to_dict()['score']
        
        return [imp_edge_index,imp_edge_score]
    def get_all_all(self,x,edge_index, node_index,top=10):
        explanation=self.Exper(x,edge_index,index=node_index)
        edge_mask=explanation.get('edge_mask')
        ans=[]
        for idx,score in enumerate(edge_mask):
            if score>0.0:
                ans.append((score,idx))
        ans=sorted(ans,key=lambda x: x[0],reverse=True)
        imp_edge_index=[i[1] for i in ans[0:top]]
        imp_edge_score=[i[0] for i in ans[0:top]]
        #return torch.Tensor(imp_edge_index),torch.Tensor(imp_edge_score)
        #return dict(zip(imp_edge_index,imp_edge_score))
        explanation=self.Exper(x,edge_index,index=node_index)
        dict_feat=explanation.visualize_feature_importance(path=None, top_k=top,visualize=False).to_dict()['score']
        
        return [imp_edge_index,imp_edge_score]