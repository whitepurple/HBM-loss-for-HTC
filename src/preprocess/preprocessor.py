from pathlib import Path
from collections import Counter
import json


class Preprocessor():
    def __init__(self, raw_dir, save_dir, hie_file):
        self.raw_dir = raw_dir
        self.save_dir = save_dir
        self.hie_file = hie_file
        self.datasets = {"train":[], "test":[], "dev":[]}
        self.parent_child_map = {} 
        self.id_label_map = {"root":"root"} 
        self.target_labels = [] 
        self.task = None

    def data_loading(self):
        pass
        
    def build_hierarchy(self):
        pass

    def save(self):
        save_dir = Path(self.save_dir)
        for split, fs in self.datasets.items():
            save_file = (save_dir/"".join([split,".jsonl"]))
            writer = save_file.open("w")
            for f in fs:
                writer.write(json.dumps(f)+"\n")
            writer.flush()
            writer.close()
            print(f"Save preprocessed {self.task} {split} dataset at {str(save_file)}.")
            print(f"\tTotal instance : {len(fs)}")

        save_file = save_dir/"parent_child_map.txt"
        writer = save_file.open("w")
        for k, v in self.parent_child_map.items():
            writer.write(f"{v}\t{k}\n")
        writer.flush()
        writer.close()

        save_file = save_dir/"labels.txt"
        writer = save_file.open("w")
        for k, v in self.id_label_map.items():
            writer.write(f"{k}\t{v}\n")
        writer.flush()
        writer.close()

        save_file = save_dir/"target_labels.txt"
        writer = save_file.open("w")
        for l in self.target_labels:
            writer.write(f"{l}\n")
        writer.flush()
        writer.close()
        
        labels = Counter()
        total_labels = Counter()
        for data in self.datasets['train']:
            l = data['labels']
            labels.update(l)
            total_labels.update(l)
            
        for data in self.datasets['dev']:
            total_labels.update(data['labels'])
            
        for data in self.datasets['test']:
            total_labels.update(data['labels'])
        
        label_freq = {'frequent':dict(),
              'few-shot':dict(),
              'zero-shot':dict(),}

        for label, freq in sorted(labels.items(), key=lambda pair: pair[1], reverse=True):
            if freq>=50:
                label_freq['frequent'][label] = freq
            else:
                label_freq['few-shot'][label] = freq
            
        zeroshot_labels = set(total_labels.keys()) - set(labels.keys())
        label_freq['zero-shot'] = {l:0 for l in zeroshot_labels}
        print([(freq,len(d)) for freq, d in label_freq.items()])
        save_file = save_dir/"label_freq.json"
        writer = save_file.open("w")
        writer.write(json.dumps(label_freq))
        writer.flush()
        writer.close()

    def preprocessing(self):
        self.data_loading()
        self.build_hierarchy()
        self.save()
        