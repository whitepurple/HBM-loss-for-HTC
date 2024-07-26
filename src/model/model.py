import torch
from torch import nn

from src.model.encoder import get_encoder
from src.model.decoder import get_decoder

class Baseline(nn.Module):
    def __init__(self, cfg, hierarchy=None):
        super().__init__()
        self.encoder = get_encoder(cfg.model.encoder)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size, 
            cfg.num_target_labels
        )
        self.dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, batch):
        outputs = self.encoder(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask']
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return {'logits':logits, 'pred_labels':logits.sigmoid()>0.5}
    
    
    def inference(self, batch):
        return self.forward(batch)
        

class HiDEC(nn.Module):
    def __init__(self, cfg, hierarchy):
        super().__init__()
        self.encoder = get_encoder(cfg.model.encoder)
        if hasattr(self.encoder, "pooler"):
            self.encoder.pooler = None
        if hasattr(self.encoder, "cls"):
            self.encoder.cls = None
        self.decoder = get_decoder(cfg.model.decoder)
        self.h = hierarchy
        
        self.hidden_dim = self.decoder.d_model
        self.label_hidden_dim = self.encoder.config.hidden_size
        self.num_labels = self.h.props['label']
        self.max_depth = self.h.props['depth']
        
        self.embeddrop = nn.Dropout(p=0.1)
        self.hierarchy_drop = nn.Dropout(p=0.1)
        
        self.label_embedding = nn.Embedding(
            self.h.props['label'], self.label_hidden_dim,
            padding_idx=self.h.props['pad_idx'],
        )
        
        # max_depth + [PAD]
        self.level_embedding = nn.Embedding(
            self.max_depth+1, self.label_hidden_dim, 
            padding_idx=self.h.props['pad_idx'],
        )

        nn.init.normal_(self.label_embedding.weight, mean=0,
                        std=self.label_hidden_dim**-0.5)
        nn.init.normal_(self.level_embedding.weight, mean=0,
                        std=self.label_hidden_dim**-0.5)
        
        self.register_buffer("LONGTENSOR", torch.LongTensor([0]))
        self.register_buffer("BOOLTENSOR", torch.BoolTensor([False]))
        self.register_buffer("ARANGE", torch.arange(256))
        
        self.collator = None
            

    def encode(self, input_ids, attention_mask):
        hs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        return hs
    
    
    def create_hierarchy_embedding(
        self, 
        tgt_input_ids, 
        tgt_level_ids
    ):
        label_embed = self.label_embedding(tgt_input_ids)
        level_embed = self.level_embedding(tgt_level_ids)
        sub_hierarchy_embed = label_embed + level_embed
        sub_hierarchy_embed = self.embeddrop(sub_hierarchy_embed)
        return sub_hierarchy_embed


    def decode(
        self,
        sequence_output,
        sub_hierarchy_embed,
        tgt_input_ids,
        tgt_mask,
        text_attention_mask
    ):
        hs = self.decoder(
            sub_hierarchy_embed,
            sequence_output, 
            tgt_key_padding_mask=tgt_input_ids==0,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=~(text_attention_mask.bool()),
        )
        return hs[-1]
    
    
    def get_unit_logits(
        self, 
        hierarchy_output,
        tgt_position,
        tgt_child_num,
        tgt_child
    ):
        hierarchy_output = hierarchy_output[tgt_position]
        hierarchy_output = hierarchy_output.repeat_interleave(tgt_child_num, dim=0)
        hierarchy_output = self.hierarchy_drop(hierarchy_output)
        
        child_embed = self.label_embedding(tgt_child)
        logits = hierarchy_output*child_embed
        logits = logits.sum(dim=-1)

        return logits 


    def forward(self, batch):
        sequence_output = self.encode(
            batch['input_ids'], 
            batch['attention_mask']
        )
        sub_hierarchy_embed = self.create_hierarchy_embedding(
            batch['tgt_input_ids'], 
            batch['tgt_level_ids']   
        )
        hierarchy_output = self.decode(
            sequence_output,
            sub_hierarchy_embed,
            batch['tgt_input_ids'],
            batch['tgt_mask'],
            batch['attention_mask']
        )
        logits = self.get_unit_logits(
            hierarchy_output,
            batch['tgt_position'],
            batch['tgt_child_num'],
            batch['tgt_child'],
        )
        return {'logits':logits}


    def _expansion(self, batch, logits, threshold=0.5):
        tgt_input_ids = batch['tgt_input_ids']
        bs = tgt_input_ids.shape[0]
        tgt_position = batch['tgt_position']
        tgt_batch_idx = self.ARANGE[:bs].repeat_interleave(tgt_position.sum(1)).cpu().tolist()
        tgt_parent = tgt_input_ids[tgt_position].cpu().tolist()
        tgt_child = batch['tgt_child']
        tgt_child_num_cpu = batch['tgt_child_num_cpu']
        expanded_nodes = batch['expanded_nodes']
        
        for bid, parent, children, unit_logit in zip(  tgt_batch_idx,
                                                tgt_parent,
                                                tgt_child.split(tgt_child_num_cpu), 
                                                logits.split(tgt_child_num_cpu)):
            pred_mask = unit_logit.sigmoid() > threshold
            child_idxs = children[pred_mask].cpu().tolist()
            child_nodes = [self.h.idx2nodes[idx] for idx in child_idxs]
            for i in range(len(child_nodes)):
                if child_nodes[i][0].is_equal(self.h.end):
                    child_nodes[i] = [n.end for n in self.h.idx2nodes[parent]]
            expanded_nodes[bid].extend(sum(child_nodes, []))
        
        
    def sub_hierarchy_expansion(self, batch, logits, level, result, threshold=0.5):
        expansion_finished = False
        self._expansion(batch, logits, threshold)
        
        bs = batch['tgt_input_ids'].shape[0]
        expanded_nodes = batch['expanded_nodes']

        render_results = []
        result['pred_node_list'] = []
        result['pred_sh_node_list'] = []
        
        for i in range(bs):
            render, source = self.h.render_for_expansion(expanded_nodes[i], stage=level+1)
            expanded_nodes[i] = source['label_nodes']
            render_results.append(render)
            result['pred_node_list'].append(render['pred_node_idxs'])
            result['pred_sh_node_list'].append([n.idx for n in source['sh_nodes']])
        
        for key in render_results[0].keys():
            batch[key] = []
        
        self.collator._collate_batch_h(batch, render_results)
        
        if len(batch['tgt_child']) == 0:
            expansion_finished = True
            return expansion_finished
            
        self.collator._to_tensor_h(batch)
        for k, v in batch.items():
            if type(v) != torch.Tensor:
                continue
            if v.type() == "torch.LongTensor":
                batch[k] = v.type_as(self.LONGTENSOR)
            elif v.type() == "torch.BoolTensor":
                batch[k] = v.type_as(self.BOOLTENSOR)
                
        return expansion_finished
    
    
    def inference(self, batch):
        result = {}
        bs = batch['input_ids'].shape[0]
        batch['expanded_nodes'] = [[] for _ in range(bs)]
        
        sequence_output = self.encode(
            batch['input_ids'], 
            batch['attention_mask']
        )
        
        for level in range(self.max_depth):
            sub_hierarchy_embed = self.create_hierarchy_embedding(
                batch['tgt_input_ids'], 
                batch['tgt_level_ids']   
            )
            hierarchy_output = self.decode(
                sequence_output,
                sub_hierarchy_embed,
                batch['tgt_input_ids'],
                batch['tgt_mask'],
                batch['attention_mask']
            )
            logits = self.get_unit_logits(
                hierarchy_output,
                batch['tgt_position'],
                batch['tgt_child_num'],
                batch['tgt_child'],
            )
            finished = self.sub_hierarchy_expansion(batch, logits, level, result)
            if finished:
                break
        return result


class HiDEC_HBM(HiDEC):
    def __init__(self, cfg, hierarchy):
        super().__init__(cfg, hierarchy)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1, bias=False)
        )
        nn.init.xavier_normal_(self.fc[0].weight)
        nn.init.zeros_(self.fc[0].bias)
        nn.init.zeros_(self.fc[2].weight)
    
    
    def get_unit_logits(
        self, 
        hierarchy_output,
        tgt_position,
        tgt_child_num,
        tgt_child
    ):
        h_output = hierarchy_output[tgt_position]
        h_output_interleaved = h_output.repeat_interleave(tgt_child_num, dim=0)
        h_output_interleaved = self.hierarchy_drop(h_output_interleaved)
        
        child_embed = self.label_embedding(tgt_child)
        logits = h_output_interleaved*child_embed
        logits = logits.sum(dim=-1)

        bound_out = h_output
        bound = self.fc(bound_out)
        
        return logits, bound
    
    
    def forward(self, batch):
        sequence_output = self.encode(
            batch['input_ids'], 
            batch['attention_mask']
        )
        sub_hierarchy_embed = self.create_hierarchy_embedding(
            batch['tgt_input_ids'], 
            batch['tgt_level_ids']   
        )
        hierarchy_output = self.decode(
            sequence_output,
            sub_hierarchy_embed,
            batch['tgt_input_ids'],
            batch['tgt_mask'],
            batch['attention_mask']
        )
        logits, bound = self.get_unit_logits(
            hierarchy_output,
            batch['tgt_position'],
            batch['tgt_child_num'],
            batch['tgt_child'],
        )
        return {'logits':logits, 'bound':bound}
    
    
    def _expansion(self, batch, logits, threshold):
        tgt_input_ids = batch['tgt_input_ids']
        bs = tgt_input_ids.shape[0]
        tgt_position = batch['tgt_position']
        tgt_batch_idx = self.ARANGE[:bs].repeat_interleave(tgt_position.sum(1)).cpu().tolist()
        tgt_parent = tgt_input_ids[tgt_position].cpu().tolist()
        tgt_child = batch['tgt_child']
        tgt_child_num_cpu = batch['tgt_child_num_cpu']
        expanded_nodes = batch['expanded_nodes']
        
        for bid, parent, children, unit_logit, th in zip(   tgt_batch_idx,
                                                        tgt_parent,
                                                        tgt_child.split(tgt_child_num_cpu), 
                                                        logits.split(tgt_child_num_cpu),
                                                        threshold.squeeze(-1)):
                pred_mask = unit_logit > th
                child_idxs = children[pred_mask].cpu().tolist()
                child_nodes = [self.h.idx2nodes[idx] for idx in child_idxs]
                for i in range(len(child_nodes)):
                    if child_nodes[i][0].is_equal(self.h.end):
                        child_nodes[i] = [n.end for n in self.h.idx2nodes[parent]]
                expanded_nodes[bid].extend(sum(child_nodes, []))
    
    
    def inference(self, batch):
        result = {}
        bs = batch['input_ids'].shape[0]
        batch['expanded_nodes'] = [[] for _ in range(bs)]
        sequence_output = self.encode(
            batch['input_ids'], 
            batch['attention_mask']
        )
        
        for level in range(self.max_depth):
            sub_hierarchy_embed = self.create_hierarchy_embedding(
                batch['tgt_input_ids'], 
                batch['tgt_level_ids']   
            )
            hierarchy_output = self.decode(
                sequence_output,
                sub_hierarchy_embed,
                batch['tgt_input_ids'],
                batch['tgt_mask'],
                batch['attention_mask']
            )
            logits, bound = self.get_unit_logits(
                hierarchy_output,
                batch['tgt_position'],
                batch['tgt_child_num'],
                batch['tgt_child'],
            )
            finished = self.sub_hierarchy_expansion(batch, logits, level, result, threshold=bound)
            if finished:
                break
        return result


model_cls_dict = {
    "baseline" : Baseline,
    "hidec" : HiDEC,
    "hidec-hbm" : HiDEC_HBM,
}


def get_model(cfg, hierarchy=None):
    return model_cls_dict[cfg.model.cls](cfg, hierarchy=hierarchy)