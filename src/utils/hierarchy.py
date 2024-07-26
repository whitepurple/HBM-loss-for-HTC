import numpy as np
from pathlib import Path
from collections import defaultdict
from enum import Enum
import random
from anytree import Node
import json


class HierarchyNode(Node):
    def __init__(self, name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.__children_ids = None
        self.__pathname = None
        self.__path = None
        self.__path_ids = None
        self.__level = None
    
    def is_equal(self, other):
        if hasattr(other, "name"):
            return self.name == other.name
        return self.name == other

    def __hash__(self):
        return hash(self.__pathname)
  
    @property
    def children_ids(self):
        if self.__children_ids is None:
            if self.children is None:
                self.__children_ids = []
            else:
                self.__children_ids = [n.idx for n in self.children]    
        return self.__children_ids
    
    @property
    def pathname(self):
        if self.__pathname is None:
            self.__pathname = self.separator.join([n.name for n in self.path])
        return self.__pathname
    
    @property
    def path(self):
        if self.__path is None:
            self.__path = list(super().path)
        return self.__path

    @property
    def path_ids(self):
        if self.__path_ids is None:
            self.__path_ids = [n.idx for n in self.path]    
        return self.__path_ids

    @property
    def level(self):
        if self.__level is None:
            self.__level = super().depth+1
        return self.__level

    @property
    def nytname(self):
        if self.children:
            return self.name.split(self.separator)[-1]
        else:
            return self.separator.join(self.pathname.split(self.separator)[-2:])

    @property
    def nytpathname(self):
        return self.separator.join([n.nytname for n in self.path])

    
class Frequent(Enum):
    NONE = -1
    COMMON = 0
    FEWSHOT = 1
    ZEROSHOT = 2
    
    
class Hierarchy:
    def __init__(
        self, 
        dataset_name,
        data_dir,
        target_labels,
        hierarchy_node,
        hierarchy_tree,
        params=None
    ):
        self.dataset_name = dataset_name
        data_dir = Path(data_dir)
        self.node_file = data_dir/hierarchy_node
        self.tree_file = data_dir/hierarchy_tree
        self.params = params
        
        self.freq = Frequent
        ## label2node : dataset label name -> anytree Node
        self.label2nodes = defaultdict(list)
        ## label2idx : dataset label name -> label idx
        self.label2idx = {"[PAD]":0, "(":1, ")":2, "[END]":3}
        self.end = HierarchyNode(
            "[END]", 
            desc="apply node to label", 
            idx=self.label2idx["[END]"], 
            is_target=False,
            end=None,
            frequent=self.freq.NONE
        )
        self.label2nodes["[END]"].append(self.end)
        sidx = max(self.label2idx.values())+1
        
        
        # create nodes
        for i, line in enumerate(self.node_file.open().readlines()):
            name, desc = line.strip().split("\t")
            idx = sidx + i
            self.label2nodes[name].append(
                HierarchyNode(
                    name, 
                    desc=desc, 
                    idx=idx,
                    is_target=False,
                    end=None,
                    frequent=self.freq.NONE
                )
            )
            self.label2idx[name] = idx

        # create adj_list 
        adj_list = defaultdict(list)
        for line in self.tree_file.open().readlines():
            parent, child = line.strip().split("\t")
            adj_list[parent].append(child)

        # perform a topological sort to get a linear ordering of the nodes
        order = []
        in_degree = defaultdict(int)
        for parent in adj_list:
            for child in adj_list[parent]:
                in_degree[child] += 1
        queue = [parent for parent in adj_list if in_degree[parent] == 0]
        
        while queue:
            parent = queue.pop(0)
            order.append(parent)
            for child in adj_list[parent]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        # construct the tree
        for parent in order:
            for child in adj_list[parent]:
                parent_nodes = self.label2nodes[parent]
                child_nodes = self.label2nodes[child]
                new_child_nodes = []
                child_node = child_nodes[0]
                for parent_node in parent_nodes:
                    if child_node.parent == None:
                        child_node.parent = parent_node
                    else:
                        new_child = HierarchyNode(
                                child_node.name, 
                                desc=child_node.desc, 
                                idx=child_node.idx,
                                is_target=False,
                                end=None,
                                frequent=child_node.frequent
                            )
                        new_child.parent = parent_node
                        new_child_nodes.append(new_child)
                child_nodes.extend(new_child_nodes)
                
        ## id2node : label idx -> anytree Nodes
        self.idx2nodes = {n[0].idx : n for n in self.label2nodes.values()}
        self.root = self.idx2nodes[sidx][0]
            
        ## id2label : label idx -> dataset label name
        self.idx2label = {i:l for l,i in self.label2idx.items()}
            
        ## target_metric_mask : dataset target label numpy bool mask
        target_labels = (data_dir/target_labels).open().readlines()
        target_labels = [l.strip() for l in target_labels]
        self.target_labels = target_labels
        
        self.target_label2idx = {}
        if dataset_name == "eurlex":
            self.target_metric_mask = np.zeros(
                (self.root.height+3+1, len(self.label2idx)), dtype=np.dtype(bool))
            ## levels + (freq, few, zero), total
            if (data_dir/"label_freq.json").exists():
                label_freq = json.load((data_dir/"label_freq.json").open())
                for l in label_freq["frequent"].keys():
                    for node in self.label2nodes[l]:
                        node.frequent = self.freq.COMMON
                    self.target_metric_mask[-4][self.label2idx[l]] = True
                
                for l in label_freq["few-shot"].keys():
                    for node in self.label2nodes[l]:
                        node.frequent = self.freq.FEWSHOT
                    self.target_metric_mask[-3][self.label2idx[l]] = True
                
                for l in label_freq["zero-shot"].keys():
                    for node in self.label2nodes[l]:
                        node.frequent = self.freq.ZEROSHOT
                    self.target_metric_mask[-2][self.label2idx[l]] = True
                    
        elif dataset_name == "bioasq":
            target_tree2nodes = {l:self.label2nodes[l] for l in target_labels}            
            
            target_desc2nodes = defaultdict(list)
            for nodes in target_tree2nodes.values():
                target_desc2nodes[nodes[0].desc].extend(nodes)
            target_desc2nodes = dict(target_desc2nodes)
            
            self.idx2target_desc = dict()
            self.desc_idx2nodes = defaultdict(list)
            
            for i, (desc, nodes) in enumerate(target_desc2nodes.items()):
                self.idx2target_desc[i] = desc
                self.desc_idx2nodes[i].extend(nodes)
            
            target_desc2idx = {v:k for k, v in self.idx2target_desc.items()}

            self.label_idx2desc_idx = {idx: target_desc2idx[nodes[0].desc]
                                    for idx, nodes in self.idx2nodes.items() if nodes[0].desc in target_desc2idx}
            self.target_metric_mask = np.zeros(
                (self.root.height+1, len(self.idx2target_desc)), dtype=np.dtype(bool))
            
        else:
            self.target_metric_mask = np.zeros(
                (self.root.height+1, len(self.label2idx)), dtype=np.dtype(bool))
        
        for n in target_labels:
            self.target_label2idx[n] = self.label2idx[n]
            for node in self.label2nodes[n]:
                node.is_target=True
                node.end = HierarchyNode(
                                self.end.name, 
                                desc=self.end.desc, 
                                idx=self.end.idx,
                                is_target=False,
                                end=None,
                                frequent=self.end.frequent,
                                parent=node                                
                            )
                if dataset_name == "bioasq":
                    idx = self.label_idx2desc_idx[node.idx]
                    self.target_metric_mask[node.depth-1][idx] = True
                    self.target_metric_mask[-1][idx] = True
                else:
                    self.target_metric_mask[node.depth-1][node.idx] = True
                    self.target_metric_mask[-1][node.idx] = True

        self.idx2target_label = {v:k for k,v in self.target_label2idx.items()}
        self.idx2leaves = {i:n for i, n in self.idx2nodes.items() if n[0].children and n[0].children[0].is_equal(self.end)}

        for nodes in self.label2nodes.values():
            for n in nodes:
                n.pathname
        
        ## label idx -> ancestor nodes, idxs
        self.idx2ancestors = {
            nodes[0].idx : tuple(node.ancestors for node in nodes)
                for nodes in self.label2nodes.values()
        }
        self.idx2aidxs = {  
            idx : sorted(set(sum([[n.idx for n in anc] for anc in ancs], [])))
                for idx, ancs in self.idx2ancestors.items()
        }
        
        ## label idx -> children nodes, idxs
        self.idx2children = {  
            nodes[0].idx : tuple(node.children for node in nodes)
                for nodes in self.label2nodes.values()
        }
        self.idx2cidxs = {  
            idx : sorted(set(sum([[n.idx for n in child] for child in children], [])))
                for idx, children in self.idx2children.items()
        }
        
        ## label idx -> sibling nodes, idxs
        self.idx2siblings = {  
            nodes[0].idx : tuple(node.siblings for node in nodes)
                for nodes in self.label2nodes.values()
        }
        self.idx2sidxs = {  
            idx : sorted(set(sum([[n.idx for n in node] for node in siblings], [])))
                for idx, siblings in self.idx2siblings.items()
        }
        
        self.label2nodes = dict(self.label2nodes)
        self.props = {}
        self.props["label"] = len(self.label2idx)
        self.props["target_label"] = len(target_labels)
        self.props["toplevel_ids"] = [self.label2idx[n.name] for n in self.root.children]
        self.props["depth"] = self.root.height+1
        self.props["pad_idx"] = 0
        
        self.stop = lambda x : not x.children or x.children[0].is_equal(self.end)
        self.attrs_train = {
            # "tgt_seq" : "name", 
            "tgt_input_ids" : "idx", 
            "tgt_level_ids" : "level", 
            "tgt_position" : lambda x,y : not self.stop(x),
            "tgt_mask" : lambda x,y : (x,x.path),
            "tgt_child" : lambda x,y : [] if self.stop(x) else x.children_ids,
            "tgt_child_num" : lambda x,y : [] if self.stop(x) else [len(x.children_ids)],
            "tgt_golden" : lambda x,y : [] if self.stop(x) else x.children,
        }
        self.attrs_inference = {
            # "tgt_seq" : "name", 
            "tgt_input_ids" : "idx", 
            "tgt_level_ids" : "level", 
            "tgt_position" : lambda x,y : False if self.stop(x) else x.depth == y,
            "tgt_mask" : lambda x,y : (x,x.path),
            "tgt_child" : lambda x,y : [] if x.depth != y or self.stop(x) else x.children_ids,
            "tgt_child_num" : lambda x,y : [] if x.depth != y or self.stop(x) else [len(x.children_ids)],
        }
        
    ## label names -> sub hierarchy nodes
    def nodes2shns(self, labels, stage="TRAIN"):
        if not labels or isinstance(labels[0], HierarchyNode):
            label_nodes = labels
        else:
            try:
                label_nodes = [self.label2nodes[label] for label in labels]
            except:
                label_nodes = [self.idx2nodes[label] for label in labels]
            label_nodes = sum(label_nodes, [])
            
        if stage == "TRAIN":
            label_nodes = [n.end for n in label_nodes]
            sh_nodes = list(set(sum([l.path for l in label_nodes],[])))
        else:
            if len(labels) == 0:
                label_nodes = [self.root]
            sh_nodes = list(set(sum([l.path for l in label_nodes],[])))
            for node in sh_nodes:
                if node.children and node.children[0].is_equal(self.end):
                    sh_nodes.append(node.end)
                    label_nodes.append(node.end)
        
        label_nodes = sorted(label_nodes, key=lambda x : x.pathname)
        sh_nodes = sorted(sh_nodes, key=lambda x : x.pathname)
        
        return {
            "sh_nodes" : sh_nodes, 
            "label_nodes" : label_nodes,
            "stage" : stage
        }

    
    def render_for_dataset(self, labels, stage="TRAIN"):
        source = self.nodes2shns(labels, stage=stage)
        return self._render_attrs(source)
    
        
    def render_for_expansion(self, labels, stage:int=0):
        source = self.nodes2shns(labels, stage=stage)
        return self._render_attrs_for_expansion(source)
    
    
    def _render_attrs(self, source):
        stage = source["stage"]
        attrnames = self.attrs_train if stage == "TRAIN" else self.attrs_inference
        
        result = {}
        for key, attrname in attrnames.items():
            get_attr = lambda node : attrname(node, stage) if callable(attrname) else getattr(node, attrname, "")
            result[key] = [get_attr(node) for node in source["sh_nodes"] if not node.is_equal(self.end)]
        result["tgt_child"] = sum(result["tgt_child"], [])
        result["tgt_child_num"] = sum(result["tgt_child_num"], [])
        
        mask = result["tgt_mask"]
        mask = [[node not in path_nodes for node, _ in mask] for _, path_nodes in mask]
        result["tgt_mask"] = mask
        if stage=="TRAIN":
            tgt_golden = [[n in source["sh_nodes"] for n in s] for s in result["tgt_golden"]]
            result["tgt_golden"] = sum(tgt_golden, [])
        return result
    
    
    def _render_attrs_for_expansion(self, source):
        result = self._render_attrs(source)
        result["pred_node_idxs"] = [n.parent.idx for n in source["sh_nodes"] 
                                    if n.name == self.end.name]
        return result, source
    
    
class HierarchyWithNaiveNeg(Hierarchy):
    def get_negative_node(self, source):
        label_ids = [node.parent.idx for node in source['label_nodes']]
        if isinstance(self.params.p, float):
            num_negs = int(len(label_ids)*self.params.p)
        else:
            num_negs = self.params.p
        neg_pools = set(self.idx2target_label.keys()) - set(label_ids)
        neg_ids = random.choices(list(neg_pools), k=max(num_negs, 1))
        return sum([self.idx2nodes[i] for i in neg_ids], [])
        
    
    def render_for_dataset(self, labels, stage="TRAIN"):
        source = self.nodes2shns(labels, stage=stage)
        if stage=="TRAIN":
            neg = self.get_negative_node(source)
            sh_nodes_with_neg = list(source["sh_nodes"])+sum([l.path for l in neg],[])
            sh_nodes_with_neg = sorted(list(set(sh_nodes_with_neg)), key=lambda x : x.pathname)
            source["sh_nodes_with_neg"] = sh_nodes_with_neg
        return self._render_attrs(source)
    
    
    def _render_attrs(self, source):
        stage = source["stage"]
        attrnames = self.attrs_train if stage=="TRAIN" else self.attrs_inference
        
        result = {}
        sub_hierarchy_nodes = source["sh_nodes"]
        if stage=="TRAIN":
            sub_hierarchy_nodes = source["sh_nodes_with_neg"]
            
        for key, attrname in attrnames.items():
            get_attr = lambda node : attrname(node, stage) if callable(attrname) else getattr(node, attrname, "")
            result[key] = [get_attr(node) for node in sub_hierarchy_nodes if not node.is_equal(self.end)]
        result["tgt_child"] = sum(result["tgt_child"], [])
        result["tgt_child_num"] = sum(result["tgt_child_num"], [])
        
        mask = result["tgt_mask"]
        mask = [[node not in path_nodes for node, _ in mask] for _, path_nodes in mask]
        result["tgt_mask"] = mask
        if stage=="TRAIN":
            tgt_golden = [[n in source["sh_nodes"] for n in s] for s in result["tgt_golden"]]
            result["tgt_golden"] = sum(tgt_golden, [])
        return result
    

class HierarchyWithFinegrainedNeg(HierarchyWithNaiveNeg):
    def get_negative_node(self, source):
        label_nodes = [node.parent for node in source['label_nodes']]
        label_node_ids = [n.idx for n in label_nodes]
        label_decendants = [sum(n.pathname in path for path in [l.pathname for l in label_nodes]) for n in label_nodes]
        num_leaves = sum(i==1 for i in label_decendants)
        if isinstance(self.params.p, float):
            num_negs = int(num_leaves*self.params.p)
        else:
            num_negs = self.params.p
        # neg_pools = set(self.idx2leaves.keys()) - set(label_node_ids)
        neg_pools = set(self.idx2target_label.keys()) - set(label_node_ids) #!
        neg_ids = random.choices(list(neg_pools), k=max(num_negs, 1))
        return sum([self.idx2nodes[i] for i in neg_ids], [])
    

hierarchy_cls_dict = {
    "default" : Hierarchy,
    "neg" : HierarchyWithNaiveNeg,
    "neg-fine" : HierarchyWithFinegrainedNeg,
}

def get_hierarchy(cfg):
    cls_name = cfg.name
    params = cfg.params
    
    dataset_name = cfg.dataset_name
    data_dir = cfg.data_dir
    target_labels = cfg.target_labels
    hierarchy_node = cfg.hierarchy_node
    hierarchy_tree = cfg.hierarchy_tree
            
    hierarchy_cls = hierarchy_cls_dict[cls_name]
    hierarchy = hierarchy_cls(
        dataset_name,
        data_dir,
        target_labels,
        hierarchy_node,
        hierarchy_tree,
        params=params
    )
    return hierarchy