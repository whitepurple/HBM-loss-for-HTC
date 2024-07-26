import torch
from collections import defaultdict
from src.utils.utils import multi_hot

class DataCollator:
    def __init__(self, num_labels: int=0):
        self.num_labels = num_labels
        self.key2type = {
            "input_ids": torch.long,
            "token_type_ids": torch.long,
            "attention_mask": torch.long,
        }

    def _apply_labels(self, batch: dict[str,list], features: list[dict[str,list[int]]]) -> None:
        """
            Function: batch dict에 label_ids, multi_hot_labels를 추가

            * Params:
                * batch: 최종 출력 batch dict
                * features : batch로 처리되기 전의 dict
        """
        for feature in features:
            batch['label_ids'].append(feature['label_ids'])
            del feature['label_ids']
        batch['multi_hot_labels'] = multi_hot(batch['label_ids'], self.num_labels)
            
    def _truncate(self, data, max_len, pad):
        return (data +[pad]*max_len)[:max_len]
        
    def _mask_truncate(self, data, max_len, pad):
        for i in range(max_len):
            try:
                data[i]+=[pad]*max_len
                data[i] = data[i][:max_len]
            except IndexError:
                data.append([pad]*max_len)
        return data

    def _collate_batch(self, batch: dict[str,list], features: list[dict[str,list[int]]]) -> None:
        input_max_len = 0
        for feature in features:
            input_max_len = max(input_max_len, len(feature["input_ids"]))
        batch['input_max_len'] = input_max_len
        for feature in features:
            self._feature_to_batch(batch, feature)
            
    def _feature_to_batch(self, batch: dict[str,list], feature: dict[str,list[int]]) -> None:
        mlen = batch['input_max_len']
        batch['input_ids'].append(self._truncate(feature['input_ids'], mlen, 0))
        batch['token_type_ids'].append(self._truncate(feature['token_type_ids'], mlen, 0))
        batch['attention_mask'].append(self._truncate(feature['attention_mask'], mlen, False))
        del feature['input_ids']
        del feature['token_type_ids']
        del feature['attention_mask']
        
    def _to_tensor(self, batch: dict[str,list]) -> None:
        for k, v in self.key2type.items():
            if not isinstance(batch[k], torch.Tensor):
                batch[k] = torch.Tensor(batch[k])
            batch[k] = batch[k].to(v)
                
    
    def __call__(self, features: list[dict]) -> dict:
        batch = defaultdict(list)
        self._apply_labels(batch, features)
        self._collate_batch(batch, features)
        self._to_tensor(batch)
        return dict(batch)
    

class HiDECDataCollator(DataCollator):
    def __init__(self, num_labels: int = 0):
        super().__init__(num_labels)
        self.key2type_h = {
                "tgt_input_ids": torch.long,
                "tgt_level_ids": torch.long,
                "tgt_position": torch.bool,
                "tgt_mask": torch.bool,
                "tgt_child": torch.long,
                "tgt_child_num": torch.long,
                "tgt_golden": torch.bool,
            }

    def _apply_labels(self, batch, features):
        super()._apply_labels(batch, features)
        if 'tgt_golden' in features[0]:
            for feature in features:
                batch['tgt_golden']+=feature['tgt_golden']
                del feature['tgt_golden']
        
        
    def _collate_batch_h(self, batch, features):
        tgt_max_len = 0
        for feature in features:
            tgt_max_len = max(tgt_max_len, len(feature["tgt_input_ids"]))
        batch['tgt_max_len'] = tgt_max_len
        for feature in features:
            self._feature_to_batch_h(batch, feature)
        
        
    def _collate_batch(self, batch, features):
        super()._collate_batch(batch, features)
        self._collate_batch_h(batch, features)
    
    
    def _feature_to_batch_h(self, batch, feature):
        mlen = batch['tgt_max_len']
        batch['tgt_input_ids'].append(self._truncate(feature['tgt_input_ids'], mlen, 0))
        batch['tgt_level_ids'].append(self._truncate(feature['tgt_level_ids'], mlen, 0))
        batch['tgt_position'].append(self._truncate(feature['tgt_position'], mlen, False))
        batch['tgt_mask'].append(self._mask_truncate(feature['tgt_mask'], mlen, False))
        batch['tgt_child']+=feature['tgt_child']
        batch['tgt_child_num']+=feature['tgt_child_num']
        
        del feature['tgt_input_ids']
        del feature['tgt_level_ids']
        del feature['tgt_position']
        del feature['tgt_mask']
        del feature['tgt_child']
        del feature['tgt_child_num']
        
        
    def _to_tensor_h(self, batch):
        batch['tgt_child_num_cpu'] = batch['tgt_child_num']
        for k, v in self.key2type_h.items():
            if not isinstance(batch[k], torch.Tensor):
                batch[k] = torch.Tensor(batch[k])
            batch[k] = batch[k].to(v)
        
    
    def _to_tensor(self, batch):
        super()._to_tensor(batch)
        self._to_tensor_h(batch)
        
    def __call__(self, features: list[dict]) -> dict:
        batch = defaultdict(list)
        self._apply_labels(batch, features)
        self._collate_batch(batch, features)
        self._to_tensor(batch)
        return dict(batch)
            

collator_cls_dict = {
    "baseline" : DataCollator,
    "hidec" : HiDECDataCollator,
}

def get_collator(collator_cls, num_labels):
    return collator_cls_dict[collator_cls](num_labels)