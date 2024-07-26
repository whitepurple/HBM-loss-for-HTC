import json
from pathlib import Path
from torch.utils.data.dataset import Dataset
from src.utils.hierarchy import Hierarchy


class ClassificationDataset(Dataset):
    def __init__(self, cfg, stage='TRAIN', **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage

        target_labels = Path(cfg.data.data_dir, cfg.data.target_labels).open().readlines()
        target_labels = [l.strip() for l in target_labels]
        self.label2idx = {label:idx for idx, label in enumerate(target_labels)}
        self.idx2label = {idx:label for label, idx in self.label2idx.items()}
        self.num_labels = len(self.label2idx)
        cache_name = f"{cfg.dataset_name}-{cfg.model.encoder.pretrained_model_name_or_path.split('/')[-1]}"
        self.cache_dir = Path(cfg.data.cache_dir, cache_name, stage)

        self.remain_chunk_files = sorted(list(self.cache_dir.iterdir()), reverse=True)
        self.chunk_count = len(self.remain_chunk_files)
        
        current_chunk_files = self.remain_chunk_files.pop()
        if not self.cache_dir.exists():
            raise Exception(f"Cache of {cache_name} is not cached. Please caching it with caching.py")
        else:
            self.data = [json.loads(line) for line in current_chunk_files.open().readlines()]
        self.chunk_size = len(self.data)
        
        last_chunk_size = self.chunk_size
        if self.chunk_count > 1:
            with self.remain_chunk_files[0].open() as f:
                last_chunk_size = len(f.readlines())
        self.total_dataset_size = self.chunk_size*(self.chunk_count-1) + last_chunk_size


    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx: int):
        if idx >= self.__len__():
            raise IndexError
        return self._preprocess_sample(idx)
    
    
    def next_chunk(self):
        if len(self.remain_chunk_files) == 0:
            self.remain_chunk_files = sorted(list(self.cache_dir.iterdir()), reverse=True)
        current_chunk_files = self.remain_chunk_files.pop()
        self.data = [json.loads(line) for line in current_chunk_files.open().readlines()]
        #! 호출 후 데이터 로더를 새로 만들어야함


    def _apply_label_list(self, sample: dict):
        sample['label_ids'] = [self.label2idx[l] for l in sample["labels"]]
        
        
    def _apply_source_preprocess(self, sample: dict):
        # token_type_ids and attention_mask for input of LM
        sample["token_type_ids"] = [0 for _ in sample["input_ids"]]
        sample["attention_mask"] = [1 for _ in sample["input_ids"]]
    
    
    def _apply_target_preprocess(self, sample: dict) -> None:
        pass


    def _preprocess_sample(self, idx: int):
        sample = self.data[idx]
        self._apply_label_list(sample)
        self._apply_source_preprocess(sample)
        self._apply_target_preprocess(sample)
        return sample


class HTCDataset(ClassificationDataset):
    def __init__(self, *args, hierarchy:Hierarchy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = hierarchy
        self.label2idx = self.h.label2idx
        self.idx2label = self.h.idx2label
        self.num_labels = len(self.label2idx)
    
    
    def _apply_target_preprocess(self, sample: dict):
        if self.stage == 'TRAIN':
            render = self.h.render_for_dataset(sample["labels"], stage=self.stage)
        else:
            render = self.h.render_for_dataset([], stage=0)
        sample.update(render)


dataset_cls_dict = {
    "baseline" : ClassificationDataset,
    "htc" : HTCDataset
}
    
def get_dataset(dataset_cls):
    return dataset_cls_dict[dataset_cls]