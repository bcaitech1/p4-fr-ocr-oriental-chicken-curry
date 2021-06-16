import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from networks.hrnet import get_hrnet
from networks.coord_conv import CoordConvNet

from data.dataset import START, PAD, END

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BottleneckBlock(nn.Module):
    """
    Dense Bottleneck Block

    It contains two convolutional layers, a 1x1 and a 3x3.
    """

    def __init__(self, input_size, growth_rate, dropout_rate=0.2, num_bn=3):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added. That is the ouput
                size of the last convolutional layer.
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(BottleneckBlock, self).__init__()
        inter_size = num_bn * growth_rate
        self.norm1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_size, inter_size, kernel_size=1, stride=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(
            inter_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    """
    Transition Block

    A transition layer reduces the number of feature maps in-between two bottleneck
    blocks.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the output
        """
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return self.pool(out)


class DenseBlock(nn.Module):
    """
    Dense block

    A dense block stacks several bottleneck blocks.
    """

    def __init__(self, input_size, growth_rate, depth, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added per bottleneck block
            depth (int): Number of bottleneck blocks
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(DenseBlock, self).__init__()
        layers = [
            BottleneckBlock(
                input_size + i * growth_rate, growth_rate, dropout_rate=dropout_rate
            )
            for i in range(depth)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask=mask, value=float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, q_channels, k_channels, head_num=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.q_channels = q_channels
        self.k_channels = k_channels
        self.head_dim = q_channels // head_num
        self.head_num = head_num

        self.q_linear = nn.Linear(q_channels, self.head_num * self.head_dim)
        self.k_linear = nn.Linear(k_channels, self.head_num * self.head_dim)
        self.v_linear = nn.Linear(k_channels, self.head_num * self.head_dim)
        self.attention = ScaledDotProductAttention(
            temperature=(self.head_num * self.head_dim) ** 0.5, dropout=dropout
        )
        self.out_linear = nn.Linear(self.head_num * self.head_dim, q_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)
        q = (
            self.q_linear(q)
            .view(b, q_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(k)
            .view(b, k_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(v)
            .view(b, v_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b, q_len, self.head_num * self.head_dim)
        )
        out = self.out_linear(out)
        out = self.dropout(out)

        return out


class Feedforward(nn.Module):
    def __init__(self, filter_size=2048, hidden_dim=512, dropout=0.1):
        super(Feedforward, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, filter_size, True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(filter_size, hidden_dim, True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )

    def forward(self, input):
        return self.layers(input)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_size, filter_size, head_num, dropout_rate=0.2):
        super(TransformerEncoderLayer, self).__init__()

        self.attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=input_size,
            head_num=head_num,
            dropout=dropout_rate,
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)
        self.feedforward_layer = Feedforward(
            filter_size=filter_size, hidden_dim=input_size
        )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, input):

        att = self.attention_layer(input, input, input)
        out = self.attention_norm(att + input)

        ff = self.feedforward_layer(out)
        out = self.feedforward_norm(ff + out)
        return out


class PositionalEncoding2D_paper(nn.Module):
    """
    2D position encoding 논문 버전
    """
    def __init__(self, in_channels, max_h=64, max_w=128, dropout=0.1):
        super(PositionalEncoding2D_paper, self).__init__()
        
        self.h_position_encoder = self.generate_encoder(in_channels, max_h) # max_len, in_channel
        self.w_position_encoder = self.generate_encoder(in_channels, max_w) # max_len, in_channel

        self.h_linear = nn.Linear(in_channels, in_channels)
        self.w_linear = nn.Linear(in_channels, in_channels)

        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)
        i = torch.arange(in_channels).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)
        position_encoder = pos * angle_rates
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])
        return position_encoder  # (Max_len, In_channel)

    def forward(self, input, alpha, beta):
        ### Require DEBUG
        b, c, h, w = input.size()
        
        # alpha, beta의 차원의 수를 늘려줌
        alpha = alpha.unsqueeze(1)
        alpha = alpha.unsqueeze(1)
        
        beta = beta.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        h_pos_encoding = (
            self.h_position_encoder[:h, :].unsqueeze(1).to(input.get_device())
        )
        h_pos_encoding = self.h_linear(h_pos_encoding)
        h_pos_encoding = h_pos_encoding.unsqueeze(0)
        h_pos_encoding = h_pos_encoding * alpha

        w_pos_encoding = (
            self.w_position_encoder[:w, :].unsqueeze(0).to(input.get_device())
        )
        w_pos_encoding = self.w_linear(w_pos_encoding)
        w_pos_encoding = w_pos_encoding.unsqueeze(0)
        w_pos_encoding = w_pos_encoding * beta

        
        pos_encoding = h_pos_encoding + w_pos_encoding

        pos_encoding = pos_encoding.permute(0, 3, 1, 2)  # [2*D, H, W]
        
        out = input + pos_encoding
        out = self.dropout(out) 

        return out


class TransformerEncoderFor2DFeatures(nn.Module):
    """
    Transformer Encoder for Image
    1) ShallowCNN : low-level visual feature identification and dimension reduction
    2) Positional Encoding : adding positional information to the visual features
    3) Transformer Encoders : self-attention layers for the 2D feature maps
    """

    def __init__(
        self,
        hrnet_cfg,
        hidden_dim,
        filter_size,
        head_num,
        layer_num,
        dropout_rate=0.1,
        checkpoint=None,
    ):
        super(TransformerEncoderFor2DFeatures, self).__init__()
        
        self.shallow_cnn = get_hrnet(hrnet_cfg)
        self.positional_encoding = PositionalEncoding2D_paper(hidden_dim)

        # for positional encoding 2d from paper
        self.adaPool = nn.AdaptiveAvgPool2d(1)
        self.linear_h_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_h_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_w_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_w_2 = nn.Linear(hidden_dim, hidden_dim) 
        self.relu = nn.ReLU(inplace=True)

        self.attention_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_dim, filter_size, head_num, dropout_rate)
                for _ in range(layer_num)
            ]
        )
        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(self, input):

        out = self.shallow_cnn(input)  # [b, c, h, w]

        # Positional Encoding from original paper
        ge = self.adaPool(out)
        ge = ge.squeeze(-1).squeeze(-1)
        alpha = torch.sigmoid(self.linear_h_2(self.relu(self.linear_h_1(ge))))
        beta = torch.sigmoid(self.linear_h_2(self.relu(self.linear_w_1(ge))))
        out = self.positional_encoding(out, alpha, beta)

        # flatten
        b, c, h, w = out.size()
        out = out.view(b, c, h * w).transpose(1, 2)  # [b, h x w, c]

        for layer in self.attention_layers:
            out = layer(out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_size, src_size, filter_size, head_num, dropout_rate=0.2):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=input_size,
            head_num=head_num,
            dropout=dropout_rate,
        )
        self.self_attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=src_size,
            head_num=head_num,
            dropout=dropout_rate,
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.feedforward_layer = Feedforward(
            filter_size=filter_size, hidden_dim=input_size
        )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, tgt, tgt_prev, src, tgt_mask):

        if tgt_prev == None:  # Train
            att = self.self_attention_layer(tgt, tgt, tgt, tgt_mask)
            out = self.self_attention_norm(att + tgt)

            att = self.attention_layer(out, src, src)
            out = self.attention_norm(att + out)

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out)
        else:
            tgt_prev = torch.cat([tgt_prev, tgt], 1)
            att = self.self_attention_layer(tgt, tgt_prev, tgt_prev, tgt_mask)
            out = self.self_attention_norm(att + tgt)

            att = self.attention_layer(out, src, src)
            out = self.attention_norm(att + out)

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out)
        return out


class PositionEncoder1D(nn.Module):
    def __init__(self, in_channels, max_len=500, dropout=0.1):
        super(PositionEncoder1D, self).__init__()

        self.position_encoder = self.generate_encoder(in_channels, max_len)
        self.position_encoder = self.position_encoder.unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)

        i = torch.arange(in_channels).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)

        position_encoder = pos * angle_rates
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])

        return position_encoder

    def forward(self, x, point=-1):
        if point == -1:
            out = x + self.position_encoder[:, : x.size(1), :].to(x.get_device())
            out = self.dropout(out)
        else:
            out = x + self.position_encoder[:, point, :].unsqueeze(1).to(x.get_device())
        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_classes,
        src_dim,
        hidden_dim,
        filter_dim,
        head_num,
        dropout_rate,
        pad_id,
        st_id,
        layer_num=1,
        beam=3,
        checkpoint=None,
    ):
        super(TransformerDecoder, self).__init__()

        self.beam = beam
        self.embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.hidden_dim = hidden_dim
        self.filter_dim = filter_dim
        self.num_classes = num_classes
        self.layer_num = layer_num

        self.pos_encoder = PositionEncoder1D(
            in_channels=hidden_dim, dropout=dropout_rate
        )

        self.attention_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    hidden_dim, src_dim, filter_dim, head_num, dropout_rate
                )
                for _ in range(layer_num)
            ]
        )
        self.generator = nn.Linear(hidden_dim, num_classes)

        self.pad_id = pad_id
        self.st_id = st_id

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def pad_mask(self, text):
        pad_mask = text == self.pad_id
        pad_mask[:, 0] = False
        pad_mask = pad_mask.unsqueeze(1)

        return pad_mask

    def order_mask(self, length):
        order_mask = torch.triu(torch.ones(length, length), diagonal=1).bool()
        order_mask = order_mask.unsqueeze(0).to(device)
        return order_mask

    def text_embedding(self, texts):
        tgt = self.embedding(texts)
        tgt *= math.sqrt(tgt.size(2))

        return tgt

    def forward(
        self, src, text, is_train=True, batch_max_length=5, teacher_forcing_ratio=1.0
    ):

        if is_train and random.random() < teacher_forcing_ratio:
            tgt = self.text_embedding(text)
            tgt = self.pos_encoder(tgt)
            tgt_mask = self.pad_mask(text) | self.order_mask(text.size(1))
            for layer in self.attention_layers:
                tgt = layer(tgt, None, src, tgt_mask)
            out = self.generator(tgt)
        else:
            out = []
            num_steps = batch_max_length - 1
            target = torch.LongTensor(src.size(0)).fill_(self.st_id).to(device) # [START] token

            if self.beam < 2:
                features = [None] * self.layer_num

                for t in range(num_steps):
                    target = target.unsqueeze(1)
                    tgt = self.text_embedding(target)
                    tgt = self.pos_encoder(tgt, point=t)
                    tgt_mask = self.order_mask(t + 1)
                    tgt_mask = tgt_mask[:, -1].unsqueeze(1)  # [1, (l+1)]
                    for l, layer in enumerate(self.attention_layers):
                        tgt = layer(tgt, features[l], src, tgt_mask)
                        features[l] = (
                            tgt if features[l] == None else torch.cat([features[l], tgt], 1)
                        )

                    _out = self.generator(tgt)  # [b, 1, c]
                    target = torch.argmax(_out[:, -1:, :], dim=-1)  # [b, 1]
                    target = target.squeeze()   # [b]
                    out.append(_out)
                
                out = torch.stack(out, dim=1).to(device)    # [b, max length, 1, class length]
                out = out.squeeze(2)    # [b, max length, class length]
            else:
                min_len = 5
                len_penalty = 1.2

                b_size = target.size(0)
                beams = [[target, [[1] for _ in range(b_size)], [None] * self.layer_num, [[] for _ in range(b_size)]]]
                tmp_out = [list() for _ in range(b_size)]

                for t in range(num_steps):
                    tmp_beams = [list() for _ in range(b_size)]

                    for target, probs, features, save_token in beams:
                        probs = torch.tensor(probs, dtype=torch.float32)
                        target = target.unsqueeze(1)
                        tgt = self.text_embedding(target)
                        tgt = self.pos_encoder(tgt, point=t)
                        tgt_mask = self.order_mask(t + 1)
                        tgt_mask = tgt_mask[:, -1].unsqueeze(1)  # [1, (l+1)]
                        for l, layer in enumerate(self.attention_layers):
                            tgt = layer(tgt, features[l], src, tgt_mask)
                            features[l] = (
                                tgt if features[l] == None else torch.cat([features[l], tgt], 1)
                            )

                        _out = self.generator(tgt)  # [b, 1, c]
                        beams_topk = torch.topk(torch.softmax(_out[:, -1:, :], dim=-1), k=self.beam, dim=-1)  # [b, k]
                        # beams_topk = torch.topk(torch.log_softmax(_out[:, -1:, :], dim=-1), k=self.beam, dim=-1)  # [b, k]
                        
                        probs = probs.to(target.device)
                        penalty = [[(1 + len(token)) ** len_penalty] for token in save_token]
                        penalty = torch.tensor(penalty).to(target.device) / ((1 + min_len) ** len_penalty)
                        # beam_probs = probs + beams_topk[0] / penalty
                        beam_probs = probs * beams_topk[0] * penalty

                        for b in range(b_size):
                            for i in range(self.beam):
                                if beams_topk[1][b][0][i] == 1:
                                    tmp_out[b].append([beam_probs[b, 0, i].cpu().item(), [o for o in save_token[b]]])
                                    break
                                
                                tmp_beams[b].append([beams_topk[1][b, 0, i], beam_probs[b, 0, i].cpu().item(), [fe.clone() for fe in features], [o for o in save_token[b]] + [beams_topk[1][b, 0, i].item()]])

                    
                    for b in range(b_size):
                        tmp_beams[b] = sorted(tmp_beams[b], key=lambda x:x[1], reverse=True)[:self.beam]


                    new_beams = []
                    if tmp_beams:
                        for i in range(self.beam):
                            tmp_tars = []
                            tmp_prob = []
                            tmp_tokens = []
                            for b in range(b_size):    
                                tmp_tars.append(tmp_beams[b][i][0])
                                tmp_prob.append([tmp_beams[b][i][1]])
                                tmp_tokens.append(tmp_beams[b][i][3])

                            new_beams.append([torch.tensor(tmp_tars).to(target.device), tmp_prob, tmp_beams[b][i][2], tmp_tokens])
                        beams = new_beams
                

                out = [sorted(t, key=lambda x: x[0], reverse=True)[0][1] for t in tmp_out]
                for o in tmp_out:
                    print(sorted(o, key=lambda x: x[0]))
        return out



class Coord_HRNET_SATRN_BEAM(nn.Module):
    def __init__(self, FLAGS, train_dataset, checkpoint=None):
        super(Coord_HRNET_SATRN_BEAM, self).__init__()

        self.encoder = TransformerEncoderFor2DFeatures(
            hrnet_cfg=FLAGS.HRNET_SATRN.config,
            hidden_dim=FLAGS.HRNET_SATRN.encoder.hidden_dim,
            filter_size=FLAGS.HRNET_SATRN.encoder.filter_dim,
            head_num=FLAGS.HRNET_SATRN.encoder.head_num,
            layer_num=FLAGS.HRNET_SATRN.encoder.layer_num,
            dropout_rate=FLAGS.dropout_rate,
        )

        self.decoder = TransformerDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.HRNET_SATRN.decoder.src_dim,
            hidden_dim=FLAGS.HRNET_SATRN.decoder.hidden_dim,
            filter_dim=FLAGS.HRNET_SATRN.decoder.filter_dim,
            head_num=FLAGS.HRNET_SATRN.decoder.head_num,
            dropout_rate=FLAGS.dropout_rate,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            layer_num=FLAGS.HRNET_SATRN.decoder.layer_num,
        )

        self.coord_conv = CoordConvNet(out_channels=16)

        self.criterion = (
            # OCRCrossEntropyLoss(eos_token=train_dataset.token_to_id[END])
            # nn.CrossEntropyLoss()
            nn.CrossEntropyLoss(ignore_index=train_dataset.token_to_id[PAD])
        )  # without ignore_index=train_dataset.token_to_id[PAD]

        if checkpoint:
            self.load_state_dict(checkpoint)

    def forward(self, input, expected, is_train, teacher_forcing_ratio):
        coord_conv = self.coord_conv(input)
        enc_result = self.encoder(coord_conv)
        dec_result = self.decoder(
            enc_result,
            expected[:, :-1],
            is_train,
            expected.size(1),
            teacher_forcing_ratio,
        )

        return dec_result
