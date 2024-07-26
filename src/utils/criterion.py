import math
from collections import defaultdict, Counter
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence as pad

INF = torch.inf

class BaseCriterion:
    def __init__(self, params=None):
        self.params = params
            
    def __call__(self, output, batch):
        pred = output['logits']
        if 'tgt_golden' in batch:
            target = batch['tgt_golden'].float()
        else:
            target = batch['multi_hot_labels'].float()
        loss = F.binary_cross_entropy_with_logits(pred, target)
        return { 
            'loss':loss 
        }
    
    
class ZLPRChildCriterion(BaseCriterion):
    def __call__(self, output, batch):
        sizes = batch['tgt_child_num_cpu']  # size of each unit (in list)
        logits = output['logits']           # model logits of each unit (concatenated)
        target = batch['tgt_golden']        # target labels of each unit (concatenated)
        logits = (1 - 2 * target) * logits          # l_neg, - l_pos
        logits_neg = logits.where(~target, -INF)    # l_neg
        logits_pos = logits.where(target, -INF)     # - l_pos
        logits_neg = pad(logits_neg.split(sizes), True, -INF)
        logits_pos = pad(logits_pos.split(sizes), True, -INF)
        zeros = torch.zeros_like(logits_neg[:, :1])
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(logits_neg, dim=-1)
        pos_loss = torch.logsumexp(logits_pos, dim=-1)
        loss = (neg_loss + pos_loss).mean()
        return { 
            'loss': loss
        }
          

class HBMChildCriterion(BaseCriterion):
    def __init__(self, params=None):
        super().__init__(params)
        self.alpha = params.alpha
        self.margin = params.margin


    def __call__(self, output, batch):
        sizes = batch['tgt_child_num_cpu']
        logits = output['logits']
        target = batch['tgt_golden']
        bound = output['bound']
        
        ### Get pos, neg logits
        logits = (1 - 2 * target) * logits          # l_neg, - l_pos
        logits_neg = logits.where(~target, -INF)    # l_neg
        logits_pos = logits.where(target, -INF)     # - l_pos
        
        ### Compute standard deviation for biases
        
        #### Neg logits with HBM
        logits_neg_padded = pad(logits_neg.split(sizes), True, -INF)
        zero_like_padded = torch.zeros_like(logits_neg_padded)
        logits_neg_margin_mask = (2*(logits_neg_padded - bound)).sigmoid() > self.margin
        neg_mask = logits_neg_padded != -INF
        neg_mask_sum = neg_mask.sum(-1)
        logits_neg_zeropad = logits_neg_padded.where(neg_mask, zero_like_padded)
        logits_neg_mean = logits_neg_zeropad.sum(-1)/neg_mask_sum
        logits_neg_zeropad = logits_neg_zeropad - logits_neg_mean.unsqueeze(-1)
        logits_neg_zeropad = logits_neg_zeropad.where(neg_mask, zero_like_padded)
        logits_neg_var = logits_neg_zeropad.square().sum(-1)/(neg_mask_sum-1).clamp_min(1.0)
        logits_neg_std = logits_neg_var.unsqueeze(-1).nan_to_num(0.0).sqrt()
        
        logits_neg = logits_neg_padded.where(logits_neg_margin_mask, -INF) - bound
        logits_neg = logits_neg + logits_neg_std.detach()*self.alpha    # l_neg - t + b_neg
        
        #### Pos logits with HBM
        logits_pos_padded = pad(logits_pos.split(sizes), True, -INF)
        logits_pos_margin_mask = (2*(logits_pos_padded + bound)).sigmoid() > self.margin
        pos_mask = logits_pos_padded != -INF
        pos_mask_sum = pos_mask.sum(-1)
        logits_pos_zeropad = logits_pos_padded.where(pos_mask, zero_like_padded)
        logits_pos_mean = logits_pos_zeropad.sum(-1)/pos_mask_sum
        logits_pos_zeropad = logits_pos_zeropad - logits_pos_mean.unsqueeze(-1)
        logits_pos_zeropad = logits_pos_zeropad.where(pos_mask, zero_like_padded)
        logits_pos_var = logits_pos_zeropad.square().sum(-1)/(pos_mask_sum-1).clamp_min(1.0)
        logits_pos_std = logits_pos_var.unsqueeze(-1).nan_to_num(0.0).sqrt()
        
        logits_pos = logits_pos_padded.where(logits_pos_margin_mask, -INF) + bound 
        logits_pos = logits_pos + logits_pos_std.detach()*self.alpha    # - l_pos + t + b_pos
        
        ### Compute loss with logsumexp
        zeros = zero_like_padded[:, :1]
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(logits_neg, dim=-1)
        pos_loss = torch.logsumexp(logits_pos, dim=-1)
        loss = (neg_loss + pos_loss).mean()
        
        return { 
            'loss': loss
        }          
        

criterion_cls_dict = {
    "bce" : BaseCriterion,
    "hidec" : BaseCriterion,
    "hidec-zlpr" : ZLPRChildCriterion,
    "hidec-hbm" : HBMChildCriterion,
}


def get_criterion(cfg):
    criterion_cfg = cfg.model.criterion
    params = None
    if hasattr(criterion_cfg, 'params'):
        params = criterion_cfg.params
        
        if 'hbm' in criterion_cfg.cls:
            if 'hpt' in cfg.model.name:
                params.alpha = 0.1
            if cfg.data.dataset == 'eurlex':
                params.margin = 0.01
        
    criterion = criterion_cls_dict[criterion_cfg.cls]
    criterion = criterion(params=params)
    return criterion