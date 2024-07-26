import torch
from src.utils.utils import multi_hot


class Postprocess:
    def __init__(self, cfg, hierarchy) -> None:
        self.h = hierarchy
        self.num_labels = cfg.num_labels
        idx2label = {idx:label for idx, label in enumerate(hierarchy.target_labels)}
        self.label_extend_ids = torch.arange(len(idx2label)).apply_(
            lambda x : hierarchy.label2idx[idx2label[x]]
        ).sort()[1]
        self.check_device = False
        self.need_rearange = True
        self.need_rearange = torch.all(self.label_extend_ids==torch.arange(len(idx2label))).item()
    
    def __call__(self, batch, output):
        if self.need_rearange:
            device = output['pred_labels'].device
            if not self.check_device:
                self.label_extend_ids = self.label_extend_ids.to(device)
                self.check_device = True
            output['pred_labels'] = output['pred_labels'].index_select(1, self.label_extend_ids)
            output['logits'] = output['logits'].index_select(1, self.label_extend_ids)
            batch['multi_hot_labels'] = batch['multi_hot_labels'].index_select(1, self.label_extend_ids)
        return output


class HiDECPostprocess(Postprocess):
    def __call__(self, batch, output):
        if 'pred_node_list' in output:
            output['pred_labels'] = multi_hot(
                output['pred_node_list'], self.num_labels
            )
        return output


postprocess_cls_dict = {
    "baseline" : Postprocess,
    "hidec" : HiDECPostprocess,
}

def get_postprocess(cfg, hierarchy):
    return postprocess_cls_dict[cfg.model.postprocess.cls](cfg, hierarchy)