import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.dataset import START, PAD
from feature_extractor import *


# sequence, Decoder
def get_pad_mask(seq, pad_idx):
    return (seq == pad_idx).unsqueeze(-2)

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            # print(mask.shape, attn.shape, v.shape)
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn)       # 第3个维度为权重
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)     # 4*21*512 ---- 4*21*8*64
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) if mask is not None else None # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Transforme_Encoder(nn.Module):
    def __init__(
            self, d_word_vec=512, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=1024, dropout=0.1, n_position=256):

        super().__init__()

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, cnn_feature, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.dropout(self.position_enc(cnn_feature))   # position embeding

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
    
class PVAM(nn.Module):
    def __init__(self,  n_dim=512, N_max_character=25, n_position=256):

        super(PVAM, self).__init__()
        self.character_len = N_max_character

        self.f0_embedding = nn.Embedding(N_max_character, n_dim)
        
        self.w0 = nn.Linear(N_max_character, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        # first linear(512,25)
        self.we = nn.Linear(n_dim, N_max_character)

        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_output):
        reading_order = torch.arange(self.character_len, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), -1)    # (S,) -> (B, S)
        reading_order_embed = self.f0_embedding(reading_order)      # b,25,512

        t = self.w0(reading_order_embed.permute(0,2,1))     # b,512,256
        t = self.active(t.permute(0,2,1) + self.wv(enc_output))     # b,256,512
        # first linear(512,25)
        attn = self.we(t)  # b,256,25

        attn = self.softmax(attn.permute(0,2,1))  # b,25,256

        g_output = torch.bmm(attn, enc_output)  # b,25,512
        return g_output


class GSRM(nn.Module):
    # global semantic reasoning module
    def __init__(self, n_dim=512, n_class=37, PAD=37-1, n_layers=4, n_position=25):

        super(GSRM, self).__init__()

        self.PAD = PAD
        self.argmax_embed = nn.Embedding(n_class, n_dim)

        self.transformer_units = Transforme_Encoder(n_layers=n_layers, n_position=n_position)      # for global context information
        # self.transformer_units = Torch_transformer_encoder(n_layers=n_layers, n_position=n_position)

    def forward(self, e_out):    
        e_argmax = e_out.argmax(dim=-1)     # b, 25
        e = self.argmax_embed(e_argmax)  # b,25,512

        e_mask = get_pad_mask(e_argmax, self.PAD)   # b,25,1
        s = self.transformer_units(e, None)   # b,25,512

        return s

class SRN_Decoder(nn.Module):
    # the wrapper of decoder layers
    def __init__(self, pad_id, n_dim=512, n_class=37, N_max_character=25, n_position=256, GSRM_layer=4): # n_dim : input 채널, # n_class: 클래스수, n_position: hidden_dim

        super(SRN_Decoder, self).__init__()
        
        self.pvam = PVAM(N_max_character=N_max_character, n_position=n_position)
        self.w_e = nn.Linear(n_dim, n_class)    # output layer

        # self.GSRM = GSRM(n_class=n_class, PAD=n_class-1, n_dim=n_dim, n_position=N_max_character, n_layers=GSRM_layer)
        self.GSRM = GSRM(n_class=n_class, PAD= pad_id, n_dim=n_dim, n_position=N_max_character, n_layers=GSRM_layer)
        self.w_s = nn.Linear(n_dim, n_class)    # output layer

        self.w_f = nn.Linear(n_dim, n_class)    # output layer

    def forward(self, cnn_feature ):
        '''cnn_feature: b,256,512 | the output from cnn'''

        g_output = self.pvam(cnn_feature)   # b,25,512
        e_out = self.w_e(g_output)     # b,25,37 ----> cross entropy loss  |  第一个输出

        s = self.GSRM(e_out)[0]      # b,25,512
        s_out = self.w_s(s)       # b,25,37f

        # TODO:change the add to gated unit
        f = g_output + s    # b,25,512
        f_out = self.w_f(f)

        return e_out, s_out, f_out

class SRN(nn.Module):
    """ SRN Model
    """
    def __init__(self,FLAGS,train_dataset,checkpoint=None):
        super(SRN, self).__init__()
        
        # Feature Extraction
        self.encoder = ResNet_FeatureExtractor(FLAGS.data.rgb)
        
        # sequence modeling
        self.seq = Transforme_Encoder(n_layers=2, n_position=231) # output의 w * h
        
        # predict
        self.decoder = SRN_Decoder(
            n_dim = FLAGS.SRN.n_dim,
            n_class = FLAGS.SRN.n_class,
            N_max_character = FLAGS.SRN.n_max_character, # max token len
            n_position = FLAGS.SRN.n_position, # output의 w*h
            pad_id=train_dataset.token_to_id[PAD]
        )
        
        self.criterion = ( # 모델 안 loss function
            nn.CrossEntropyLoss()
        )
        
        if checkpoint:
            self.load_state_dict(checkpoint)
        
    def forward(self,input):
        """Feature Extraction"""
        out = self.encoder(input)
        b, c, h, w = out.size()
        out = out.view(b, c, h * w).transpose(1, 2)
        
        """sequence process"""
        out = self.seq(out,src_mask=None)[0]
        
        output = self.decoder(out)
        
        return output[2]
