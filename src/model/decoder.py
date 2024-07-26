from torch import nn

class Decoder(nn.Module):
    def __init__(
        self, 
        d_model=768, 
        nhead=12, 
        dim_feedforward=3072, 
        dropout=0.1, 
        activation="relu",
        num_layers=2, 
    ):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers, 
            norm=decoder_norm,
        )
        self.d_model = d_model
        self.nhead = nhead


    def forward(
        self, 
        query_embed, 
        memory, 
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        bs = query_embed.shape[0]    
        if tgt_mask.shape[0] != bs*self.nhead:
            tgt_mask = tgt_mask.repeat_interleave(self.nhead,0)
        assert tgt_mask.shape[0] == bs*self.nhead
        hs = self.decoder(
            query_embed, 
            memory, 
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        ).unsqueeze(0)
        #* torch.Size([#layer, query_length, batch_size, d_model])
        #* torch.Size([1, 10, 64, 768])
        return hs
        

def get_decoder(cfg):
    return Decoder(**cfg)
