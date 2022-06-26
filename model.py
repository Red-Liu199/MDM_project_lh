import torch
import torch.nn as nn
from ConvLSTM import ConvLSTM
from config import global_config as cfg
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        subnets=[
            OrderedDict({'conv1_leaky_1': [cfg.input_channels, 8, 3, 2, 1]}),
            OrderedDict({'conv2_leaky_1': [16, cfg.middle_channels1, 3, 1, 1]}),
            OrderedDict({'conv3_leaky_1': [cfg.middle_channels1, cfg.middle_channels2, 3, 1, 1]}),
        ]
        rnns=[
            ConvLSTM(input_channel=8, num_filter=16, b_h_w=(cfg.batch_size, 22, 28),
                 kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=cfg.middle_channels1, num_filter=cfg.middle_channels1, b_h_w=(cfg.batch_size, 22, 28),
                    kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=cfg.middle_channels2, num_filter=cfg.middle_channels2, b_h_w=(cfg.batch_size, 22, 28),
                    kernel_size=3, stride=1, padding=1)
        ]
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        outputs_stage, state_stage = rnn(inputs=input, seq_len=cfg.in_len)

        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input):
        hidden_states = []
        for i in range(1, self.blocks+1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        subnets=[
            OrderedDict({'deconv1_leaky_1': [cfg.middle_channels2, cfg.middle_channels1, 3, 1, 1]}),
            OrderedDict({'deconv2_leaky_1': [cfg.middle_channels1, 32, 3, 1, 1]}),
            OrderedDict({
                'deconv3_leaky_1': [16, 8, (3,4), 2, 1],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, cfg.output_channels, 1, 1, 0]
            }),
        ]
        rnns=[
            ConvLSTM(input_channel=cfg.middle_channels2, num_filter=cfg.middle_channels2, b_h_w=(cfg.batch_size, 22, 28),
                    kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=cfg.middle_channels1, num_filter=cfg.middle_channels1, b_h_w=(cfg.batch_size, 22, 28),
                    kernel_size=3, stride=1, padding=1),
            ConvLSTM(input_channel=32, num_filter=16, b_h_w=(cfg.batch_size, 22, 28),
                    kernel_size=3, stride=1, padding=1),
        ]

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state, seq_len=cfg.out_len)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))

        return input

        # input: 5D S*B*I*H*W

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'),
                                      getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)),
                                                       getattr(self, 'rnn' + str(i)))
        return input

class ConvLSTM_net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()

    def forward(self, input):
        # input: [S_in, B, C_in, H, W]
        # output: [S_out, B, C_out, H, W]
        state = self.encoder(input)
        output = self.decoder(state)
        return output

# transformer layers
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        # 用Linear来做投影矩阵    

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4) # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) #[B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output 

class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        
        # Temporal embedding One hot
        self.time_num = time_num
#         self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding选用one-hot方式 或者
        self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding选用nn.Embedding


        
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t):
        B, N, T, C = query.shape
        
        D_T = self.temporal_embedding(torch.arange(0, T).to(cfg.device))    # temporal embedding选用nn.Embedding
        D_T = D_T.expand(B, N, T, C)


        # temporal embedding加到query。 原论文采用concatenated
        query = query + D_T  
        
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(STTransformerBlock, self).__init__()
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, t):
        # value,  key, query: [N, T, C] [B, N, T, C]
        x = self.dropout( self.norm(self.TTransformer(value, key, query, t) + query) ) #(B, N, T, C)
        return x

class TEncoder(nn.Module):
    # 堆叠多层 Transformer Block
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        time_num,
        device,
        forward_expansion,
        dropout,
    ):

        super(TEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    embed_size,
                    heads,
                    time_num,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
    # x: [N, T, C]  [B, N, T, C]
        out = self.dropout(x) 
        for layer in self.layers:
            out = layer(out, out, out, t)
        return out     
    


### Transformer   
class Transformer(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        time_num,
        forward_expansion,
        dropout,
        
        device=cfg.device
    ):
        super(Transformer, self).__init__()
        self.encoder = TEncoder(
            embed_size,
            num_layers,
            heads,
            time_num,
            device,
            forward_expansion,
            dropout
        )
        self.device = device

    def forward(self, src, t): 
        ## scr: [N, T, C]   [B, N, T, C]
        enc_src = self.encoder(src, t) 
        return enc_src # [B, N, T, C]

### ST Transformer: Total Model

class STTransformer(nn.Module):
    def __init__(
        self, 
        in_channels, 
        embed_size, 
        time_num,
        num_layers,
        T_dim,
        output_T_dim,  
        heads,    
        forward_expansion,
        dropout = 0
    ):        
        super(STTransformer, self).__init__()

        self.forward_expansion = forward_expansion
        # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.Transformer = Transformer(
            embed_size, 
            num_layers, 
            heads, 
            time_num,
            forward_expansion,
            dropout = 0
        )

        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)  
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # input: [S_in, B, C_in, H, W]
        # output: [S_out, B, C_out, H, W]
        S, B, C, H, W = x.shape
        x=torch.reshape(x, (S, B, C, -1))# S, B, C, H*W
        x=x.permute(1, 2, 3, 0) # [B, C, N, T] 
        
        
        input_Transformer = self.conv1(x)
        input_Transformer = input_Transformer.permute(0, 2, 3, 1) # B, N, T, C
        output_Transformer = self.Transformer(input_Transformer, self.forward_expansion)  # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3) #output_Transformer shape[B, T, N, C]
        
        out = self.relu(self.conv2(output_Transformer))    #  out shape: [1, output_T_dim, N, C]        
        out = out.permute(0, 3, 2, 1)           #  out shape: [B, C, N, output_T_dim]
        out = self.conv3(out)                   #  out shape: [B, 1, N, output_T_dim]
        out=out.permute(3, 0, 1, 2) # S_out, B, 1, N
        out=torch.reshape(out, (-1, B, 1, H, W)) # S_out, B, 1, H, W
        return out

