import torch
import torch.nn as nn
from transformers import AutoModel


def get_encoder(cfg):
    return AutoModel.from_pretrained(**cfg)