import logging
from pathlib import Path
import tarfile
import shutil
import argparse
import xml.dom.minidom

from tqdm import tqdm

import tarfile
import shutil

from preprocessor import Preprocessor


class NYTPreprocessor(Preprocessor):
    def __init__(self, raw_dir, save_dir, hie_file):
        super().__init__(raw_dir, save_dir, hie_file)
        self.task = "nyt"

    def extract_data(self):
        initial_tgz = Path(self.raw_dir)/"nyt_corpus_LDC2008T19.tgz"
        extracted_dir = Path(self.raw_dir)/"nyt_corpus"
        data_dir = extracted_dir/"data"
        base_xml_directory = Path(self.raw_dir)/"Nytimes"

        if not initial_tgz.exists():
            logging.error(f"nyt_corpus_LDC2008T19.tgz doesn't exist in raw_dir. Check file name")

        if not extracted_dir.exists():
            logging.info(f"Extract the initial nyt_corpus_LDC2008T19.tgz")
            with tarfile.open(initial_tgz, 'r:gz') as tar_ref:
                tar_ref.extractall(path=self.raw_dir)
        else:
            logging.info("nyt_corpus_LDC2008T19.tgz are already Extracted")

        year_dirs = [x for x in data_dir.iterdir() if x.is_dir()]

        if not base_xml_directory.exists():
            for year_dir in tqdm(year_dirs, desc="Collecting .xml"):
                xml_year_dir = base_xml_directory / year_dir.name
                xml_year_dir.mkdir(parents=True, exist_ok=True)

                month_tgzs = [x for x in year_dir.iterdir() if x.suffix == '.tgz']
                for month_tgz in month_tgzs:
                        temp_dir = year_dir / 'temp'
                        temp_dir.mkdir(exist_ok=True)

                        with tarfile.open(month_tgz, 'r:gz') as tar_ref:
                            tar_ref.extractall(path=temp_dir)

                        month_dir_path = next(temp_dir.iterdir())
                        
                        for day_dir in month_dir_path.iterdir():
                            if day_dir.is_dir():
                                for file in day_dir.iterdir():
                                    if file.suffix == '.xml':
                                        shutil.move(str(file), xml_year_dir)
                        
                        shutil.rmtree(temp_dir)
        else:
            logging.info(f"*.xml files are already exists on {base_xml_directory}.")

    def data_loading(self):
        jobs = {"train":"idnewnyt_train.json",
                "test":"idnewnyt_test.json",
                "dev":"idnewnyt_val.json"}
        
        logging.info(f"Loading Labels from nyt_label.vocab")

        with (Path(self.raw_dir)/"nyt_label.vocab").open() as f:
            label_vocab_s = f.readlines()
        self.label_vocab = [label.strip() for label in label_vocab_s]

        for job, id_json in jobs.items():
            with (Path(self.raw_dir)/id_json).open() as f:
                ids = f.readlines()

            for file_name in tqdm(ids, desc=job.upper()):
                xml_path = file_name.strip()
                xml_path = Path(self.raw_dir)/xml_path
                did = (xml_path.name).split(".")[0]
                xml_path = str(xml_path)
                try:
                    sample = {}
                    sample["did"] = did
                    dom = xml.dom.minidom.parse(xml_path)
                    root = dom.documentElement
                    tags = root.getElementsByTagName('p')
                    text = ''
                    for tag in tags[1:]:
                        text += tag.firstChild.data
                    if text == '':
                        continue
                    sample['text'] = text
                    sample_label = []
                    tags = root.getElementsByTagName('classifier')
                    for tag in tags:
                        type = tag.getAttribute('type')
                        if type != 'taxonomic_classifier':
                            continue
                        hier_path = tag.firstChild.data
                        hier_list = hier_path.split('/')
                        if len(hier_list) < 3:
                            continue
                        for l in range(1, len(hier_list) + 1):
                            label = '/'.join(hier_list[:l])
                            if label == 'Top':
                                continue
                            if label not in sample_label and label in self.label_vocab:
                                sample_label.append(label)
                    sample['labels'] = sample_label
                    self.datasets[job].append(sample)
                except Exception as e:
                    print(e)
                    print(xml_path)
                    print('Something went wrong...')
                    continue

        logging.info('There are %s train docs', len(self.datasets["train"]))
        logging.info('There are %s val docs', len(self.datasets["dev"]))
        logging.info('There are %s test docs\n', len(self.datasets["test"]))
    
    def build_hierarchy(self):
        for label in self.label_vocab:
            if label == "Root":
                self.id_label_map["root"] = "root"
                continue
            
            splited_label = label.split("/")
            if len(splited_label) == 2:
                parent = "root"
            else:
                parent = "/".join(splited_label[:-1])
            
            self.id_label_map[label] = label
            self.target_labels.append(label)
            self.parent_child_map[label] = parent
