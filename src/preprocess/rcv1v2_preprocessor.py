import logging
from pathlib import Path
import tarfile
import gzip
import shutil
from urllib import request
from collections import defaultdict, Counter
import argparse
import xml.etree.ElementTree as ET
import xml
import re
from random import shuffle

from tqdm import tqdm

from preprocessor import Preprocessor


class RCV1v2Preprocessor(Preprocessor):
    def __init__(self, raw_dir, hie_file, save_dir) -> None:
        super().__init__(raw_dir, hie_file, save_dir)
        self.task = "rcv1v2"
        self.root = Path(raw_dir)
        logging.info('Downloading rcv1.topics.txt')
        self.might_download_file('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a01-list-of-topics/rcv1.topics.txt')
        logging.info('Downloading rcv1.topics.hier.orig')
        self.might_download_file('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig')
        logging.info('Downloading rcv1v2-ids.dat.gz')
        self.might_download_file('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a07-rcv1-doc-ids/rcv1v2-ids.dat.gz')
        self.might_extract_gz(self.root / 'rcv1v2-ids.dat.gz')
        logging.info('Downloading rcv1-v2.topics.qrels.gz')
        self.might_download_file('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz')
        self.might_extract_gz(self.root / 'rcv1-v2.topics.qrels.gz')
        logging.info('Downloading lyrl2004_tokens_train.dat.gz')
        self.might_download_file('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz')
        self.might_extract_gz(self.root / 'lyrl2004_tokens_train.dat.gz')

        logging.info('Extracting main dataset from rcv1.tar.xz')
        self.might_extract_tar(self.root / 'rcv1.tar.xz')
        

    def might_extract_tar(self, path):
        path = Path(path)
        dir_name = '.'.join(path.name.split('.')[:-2])
        dir_output = path.parent/dir_name
        if not dir_output.exists():
            if path.exists():
                tf = tarfile.open(str(path))
                tf.extractall(path.parent)
            else:
                logging.error('File %s is required. \n', path.name)


    def might_extract_gz(self, path):
        path = Path(path)
        file_output_name = '.'.join(path.name.split('.')[:-1])
        file_name = path.name
        if not (path.parent/file_output_name).exists():
            logging.info('Extracting %s ...\n', file_name)

            with gzip.open(str(path), 'rb') as f_in:
                with open(str(path.parent/file_output_name), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


    def might_download_file(self, url):
        file_name = url.split('/')[-1]
        file = self.root/file_name
        if not file.exists():
            logging.info('File %s does not exist. Downloading ...\n', file_name)
            file_data = request.urlopen(url)
            data_to_write = file_data.read()

            with file.open('wb') as f:
                f.write(data_to_write)
        else:
            logging.info('File %s already existed.\n', file_name)


    def get_doc_ids_v2(self):
        file = self.root/'rcv1v2-ids.dat'
        with file.open('rt', encoding='ascii') as i_f:
            doc_ids = i_f.readlines()
        doc_ids = [item[:-1] for item in doc_ids]
        logging.info('There are %s docs in RCV1-v2\n', len(doc_ids))
        return doc_ids


    def get_doc_topics_mapping(self):
        file = self.root / 'rcv1-v2.topics.qrels'
        with file.open('rt', encoding='ascii') as i_f:
            lines = i_f.readlines()
        lines = [item[:-1] for item in lines]
        doc_topics = defaultdict(list)
        for item in lines:
            topic, doc_id, _ = item.split()
            doc_topics[doc_id].append(topic)
        logging.info('Mapping dictionary contains %s docs\n', len(doc_topics))
        return doc_topics


    def get_topic_desc(self):
        file = self.root / 'rcv1'/'codes'/'topic_codes.txt'
        with file.open('rt', encoding='ascii') as i_f:
            lines = i_f.readlines()
        lines = [item[:-1] for item in lines if len(item)>1][2:]
        topic_desc = [item.split('\t') for item in lines]
        topic_desc = {item[0]:item[1] for item in topic_desc}

        logging.info('There are totally %s topics\n', len(topic_desc))
        return topic_desc


    def get_train_doc_ids(self):
        file = self.root / 'lyrl2004_tokens_train.dat'
        with file.open('rt', encoding='ascii') as i_f:
            line = i_f.read()
        train_doc_ids = [l.split("\n")[0][3:] for l in line.split("\n\n")[:-1]]
        return train_doc_ids


    def get_parent_child_mapping(self):
        file = self.root / 'rcv1.topics.hier.orig'
        with file.open('rt', encoding='ascii') as i_f:
            pcmap_lines = i_f.readlines()
        return pcmap_lines


    def fetch_docs(self, doc_ids):
        all_path_docs = list(self.root.glob('rcv1/199*/*'))
        docid2topics = self.get_doc_topics_mapping()

        docid2path = {p.name[:-10]:p for p in all_path_docs}
        for idx, doc_id in enumerate(doc_ids):
            # text = docid2path[doc_id].open('rt', encoding='iso-8859-1').read()
            tree = ET.parse(str(docid2path[doc_id]))
            root = tree.getroot()
            text = xml.etree.ElementTree.tostring(root, encoding='unicode')
            label = docid2topics[doc_id]
            if idx % 100000 == 0:
                logging.info('Fetched %s/%s docs', idx, len(doc_ids))
            yield doc_id, text, label, str(docid2path[doc_id])
    
    
    def data_loading(self):
        docs_ids = self.get_doc_ids_v2()
        docs = list(self.fetch_docs(docs_ids))
        train_doc_ids = self.get_train_doc_ids()
        for doc in tqdm(docs):
            xml_root = xml.etree.ElementTree.XML(doc[1])
            title = [xml_root.find("title").text]
            body = [l.text for l in xml_root.find('text').findall("p")]
            text = " ".join(title+body)
            instance = {"did":doc[0], "text":text, "labels":doc[2]}

            if doc[0] in train_doc_ids:
                self.datasets["train"].append(instance)
            else:
                self.datasets["test"].append(instance)
                
        shuffle(self.datasets["train"])
        self.datasets["dev"] = self.datasets["train"][20833:]
        self.datasets["train"] = self.datasets["train"][:20833]

        logging.info('There are %s train docs', len(self.datasets["train"]))
        logging.info('There are %s val docs', len(self.datasets["dev"]))
        logging.info('There are %s test docs\n', len(self.datasets["test"]))
    
    
    def build_hierarchy(self):
        pcmap_lines = self.get_parent_child_mapping()
        for line in pcmap_lines:
            parent_match = re.search(r"parent:\s([^\s]+)", line)
            child_match = re.search(r"child:\s([^\s]+)", line)
            description_match = re.search(r"child-description:\s(.+)", line)
            
            # Extracting values
            parent = parent_match.group(1) if parent_match else None
            child = child_match.group(1) if child_match else None
            description = description_match.group(1) if description_match else None
            
            if child == "Root":
                continue
            else:
                if parent == "Root":
                    self.parent_child_map[child] = "root"
                else:
                    self.parent_child_map[child] = parent
                self.id_label_map[child] = description
                self.target_labels.append(child)
