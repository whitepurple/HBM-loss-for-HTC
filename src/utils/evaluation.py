from functools import partial

import torch
from torchmetrics import Precision, Recall, F1Score
from torchmetrics.classification import MultilabelStatScores, MultilabelConfusionMatrix
from torchmetrics.functional.classification.precision_recall import _precision_recall_reduce
from torchmetrics.functional.classification.f_beta import _fbeta_reduce
from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg

_compute_micro_precision = partial(_precision_recall_reduce, "precision", average="micro")
_compute_micro_recall = partial(_precision_recall_reduce, "recall", average="micro")
_compute_micro_f1 = partial(_fbeta_reduce, beta=1.0, average="micro")
_compute_macro_precision = partial(_precision_recall_reduce, "precision", average="macro")
_compute_macro_precision = partial(_precision_recall_reduce, "recall", average="macro")
_compute_macro_f1 = partial(_fbeta_reduce, beta=1.0, average="macro")
_compute_ndcg = retrieval_normalized_dcg

def compute_macro_f1(tp,fp,tn,fn):
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    macro_f1 = 2 * (precision * recall) / (precision + recall + eps)
    return macro_f1.mean()

def compute_micro_f1(tp,fp,tn,fn):
    eps = 1e-12
    precision = tp.sum() / (tp.sum() + fp.sum() + eps)
    recall = tp.sum() / (tp.sum() + fn.sum() + eps)
    micro_f1 = 2 * (precision * recall) / (precision + recall + eps)
    return micro_f1.mean()


class EvaluationTorchMetric:
    def __init__(self, target_metric_mask):
        self.target_metric_mask = target_metric_mask
        self.label_nums = target_metric_mask.sum(-1).tolist()
        self.precision = Precision(task="multilabel", average='micro', num_labels=self.label_nums[-1])
        self.recall = Recall(task="multilabel", average='micro', num_labels=self.label_nums[-1])
        self.micro_f1s = [F1Score(task="multilabel", 
                                  average='micro', 
                                  num_labels=num_labels)
                          if num_labels>1 else
                          lambda x,y:torch.tensor(0.0)
                          for num_labels in self.label_nums]
        self.macro_f1 = F1Score(task="multilabel", average='macro', num_labels=self.label_nums[-1])
        self.device_checked = False
    
    def check_device(self, device):
        self.precision = self.precision.to(device)
        self.recall = self.recall.to(device)
        self.micro_f1s = [m.to(device) if hasattr(m,'to') else m for m in self.micro_f1s ]
        self.macro_f1 = self.macro_f1.to(device)
        self.device_checked = True
        
    def __call__(self, need_metrics):
        labels = need_metrics['labels']
        preds = need_metrics['pred_labels'].to(labels.device)
        bs = labels.shape[0]
        if not self.device_checked:
            self.check_device(labels.device)
        if labels.shape[-1] == self.label_nums[-1]:
            new_labels = torch.zeros(bs,self.target_metric_mask.shape[-1]).to(labels)
            new_labels[:,self.target_metric_mask[-1]] = labels
            labels = new_labels
            new_preds = torch.zeros(bs,self.target_metric_mask.shape[-1]).to(preds)
            new_preds[:,self.target_metric_mask[-1]] = preds
            preds = new_preds
            
        total_labels = labels[:, self.target_metric_mask[-1]]
        total_preds = preds[:, self.target_metric_mask[-1]]
            
        precision = self.precision(total_preds, total_labels).item()
        recall = self.recall(total_preds, total_labels).item()
        macro_f1 = self.macro_f1(total_preds, total_labels).item()
        micro_f1s = []
        for fn, mask in zip(self.micro_f1s, self.target_metric_mask):
            level_labels = labels[:, mask]
            level_preds = preds[:, mask]
            micro_f1s.append(fn(level_labels, level_preds).item())
            
        result = {
            'precision': precision,
            'recall': recall,
            'micro_f1': micro_f1s[-1],
            'macro_f1': macro_f1
        }
        for level, mif1 in enumerate(micro_f1s[:-1]):
            if level==0:
                result[f'top_micro_f1'] = mif1
            else:    
                result[f'level{level+1}_micro_f1'] = mif1
        
        return result
    
    
class EvaluationConfusionMatrix:
    def __init__(self, dataset_name, target_metric_mask, 
                    precision_k=[8,15],
                    rprecision_k=5,
                    ndcg_k=5
                 ):
        self.dataset_name = dataset_name
        self.target_metric_mask = target_metric_mask
        self.label_nums = target_metric_mask.sum(-1).tolist()
        self.num_target_labels = self.label_nums[-1]
        self.num_nodes = target_metric_mask.shape[-1]
        self._confmat = MultilabelConfusionMatrix(num_labels=self.num_nodes)
        self._device_checked = False
        self.precision_k = precision_k
        self.rprecision_k = rprecision_k
        self.ndcg_k = ndcg_k
        self.reset()
        
        
    def _check_device(self, device):
        self._confmat = self._confmat.to(device)
        self._device_checked = True


    def _safe_divide(self, num: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        denom[denom == 0.0] = 1
        return num / denom
          
          
    def reset(self):
        self._confmat.reset()    
        
        ## EMR (ExactMatchRatio)
        self._num_examples = 0
        self._num_exact_matches = 0
        
        ## R-precision (PrecisionAtRecall)
        self._rprecision_sum = 0
        
        ## R-precision@k (in EURLEX57K)
        self._rprecisionatk_sum = 0
        
        ## Precision@k
        self._precisionk_sum = [0 for _ in self.precision_k]
        
        ## MAP (MeanAveragePrecision)
        self._average_precision_sum = 0
        
        ## nDCG
        self._ndcg_sum = 0
    
    
    def update(self, pred, target) -> None:
        if not self._device_checked:
            self._check_device(target.device)
        if pred.device != target.device:
            pred = pred.to(target.device)
            
        bs, ps = pred.shape
        if ps != self.num_nodes:
            pred_extended = torch.zeros(bs, self.num_nodes, device=pred.device).bool()
            pred_extended[:,self.target_metric_mask[-1]] = pred.bool()
            pred = pred_extended
        if target.shape[-1] != self.num_nodes:
            target_extended = torch.zeros(bs, self.num_nodes, device=target.device).bool()
            target_extended[:,self.target_metric_mask[-1]] = target.bool()
            target = target_extended

        self._confmat.update(pred, target)
        
        pred = pred[:,self.target_metric_mask[-1]].long()
        target = target[:,self.target_metric_mask[-1]].long()
        
        ## EMR
        self._num_examples += bs
        self._num_exact_matches += torch.all(
            torch.eq(pred, target), dim=-1
        ).sum()
    
        ## R-precision (PrecisionAtRecall)
        num_targets = target.sum(dim=1, dtype=torch.int64)
        _, indices = torch.sort(pred, dim=1, descending=True)
        sorted_targets = target.gather(1, indices)
        sorted_targets_cum = torch.cumsum(sorted_targets, dim=1)
        self._rprecision_sum += torch.sum(
            sorted_targets_cum.gather(1, num_targets.unsqueeze(1) - 1).squeeze()
            / num_targets
        )
        
        ## R-precision@k (in EURLEX57K)
        num_targets_k = num_targets.clamp_max(self.rprecision_k)
        self._rprecisionatk_sum += torch.sum(
            sorted_targets_cum.gather(1, num_targets_k.unsqueeze(1) - 1).squeeze()
            / num_targets_k
        )

        ## Precision@k
        for i, k in enumerate(self.precision_k):
            top_k = torch.topk(pred, dim=1, k=k)
            targets_k = target.gather(1, top_k.indices)
            logits_k = torch.ones(targets_k.shape, device=targets_k.device)
            tp_state = logits_k * targets_k
            fp_state = (logits_k) * (1 - targets_k)
            tp = torch.sum(tp_state, dim=1)
            fp = torch.sum(fp_state, dim=1)
            self._precisionk_sum[i] += self._safe_divide(tp,tp + fp).sum()
            
        ## MAP
        denom = torch.arange(1, pred.shape[1] + 1, device=target.device).repeat(bs, 1)
        prec_at_k = sorted_targets_cum / denom
        average_precision_batch = torch.sum(
            prec_at_k * sorted_targets, dim=1
        ) / torch.sum(sorted_targets, dim=1)
        self._average_precision_sum += torch.sum(average_precision_batch)
        
        ## nDCG
        for p, t in zip(pred.float(), target):
            self._ndcg_sum += _compute_ndcg(p, t, top_k=self.ndcg_k)
        
    
    def get_data(self):
        return { 
                'confmat' : self._confmat.confmat,
                '_num_examples' : self._num_examples,
                '_num_exact_matches' : self._num_exact_matches,
                '_rprecision_sum' : self._rprecision_sum,
                '_rprecisionatk_sum' : self._rprecisionatk_sum,                
                '_precisionk_sum' : self._precisionk_sum,
                '_average_precision_sum' : self._average_precision_sum,
                '_ndcg_sum' : self._ndcg_sum,                
                }
    
    
    def set_data(self, datas):
        self._confmat.confmat = datas['confmat']
        self._num_examples = datas['_num_examples']
        self._num_exact_matches = datas['_num_exact_matches']
        self._rprecision_sum = datas['_rprecision_sum']
        self._rprecisionatk_sum = datas['_rprecisionatk_sum']
        self._precisionk_sum = datas['_precisionk_sum']
        self._average_precision_sum = datas['_average_precision_sum']
        self._ndcg_sum = datas['_ndcg_sum']
        
    
    def compute(self):
        target_mask = self.target_metric_mask[-1]
        ## arange confmat to tp,fp,tn,fn
        final_state = self._confmat.confmat.view(-1,4).transpose(1,0)[(3,1,0,2),:].double()
        tp,fp,tn,fn = final_state[:,target_mask]
        micro_precision = _compute_micro_precision(tp,fp,tn,fn)
        micro_recall = _compute_micro_recall(tp,fp,tn,fn)
        micro_f1 = compute_micro_f1(tp,fp,tn,fn)
        macro_f1 = compute_macro_f1(tp,fp,tn,fn)
        
        result = {
            'precision': micro_precision,
            'recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1
        }
        
        micro_f1s = []
        macro_f1s = []
        for level_mask in self.target_metric_mask[:-1]:
            micro_f1s.append(compute_micro_f1(*final_state[:,level_mask]))
            macro_f1s.append(compute_macro_f1(*final_state[:,level_mask]))
        
        if self.dataset_name == 'eurlex':
            for level, f1_score in enumerate(micro_f1s[:-3]):
                result[f'level{level+1}_micro_f1'] = f1_score
            for level, f1_score in enumerate(macro_f1s[:-3]):
                result[f'level{level+1}_macro_f1'] = f1_score
            result[f'frequent_micro_f1'] = micro_f1s[-3]
            result[f'fewshot_micro_f1'] = micro_f1s[-2]
            result[f'zeroshot_micro_f1'] = micro_f1s[-1]
            result[f'frequent_macro_f1'] = macro_f1s[-3]
            result[f'fewshot_macro_f1'] = macro_f1s[-2]
            result[f'zeroshot_macro_f1'] = macro_f1s[-1]
        else:
            for level, f1_score in enumerate(micro_f1s):
                result[f'level{level+1}_micro_f1'] = f1_score
            for level, f1_score in enumerate(macro_f1s):
                result[f'level{level+1}_macro_f1'] = f1_score
        
        result['EMR'] = self._num_exact_matches / self._num_examples
        result['R-Precision'] = self._rprecision_sum / self._num_examples
        result[f'R-Precision@{self.rprecision_k}'] = self._rprecisionatk_sum / self._num_examples
        
        for i, k in enumerate(self.precision_k):
            result[f'Precision@{k}'] = self._precisionk_sum[i] / self._num_examples
        result['MAP'] = self._average_precision_sum / self._num_examples
        result[f'nDCG@{self.ndcg_k}'] = self._ndcg_sum / self._num_examples
        
        return result
