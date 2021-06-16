import math
import random
import torch
import timm

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.dataset import START, PAD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Coordinates Convolution layer
class AddCoordinates(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, y_dim, x_dim = input_tensor.size()

        xx_ones = torch.ones([1, 1, 1, x_dim], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, y_dim], dtype=torch.int32)

        xx_range = torch.arange(y_dim, dtype=torch.int32)
        yy_range = torch.arange(x_dim, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (y_dim - 1)
        yy_channel = yy_channel.float() / (x_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConvNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=1,
                 stride=1, padding=0, with_r=False):
        super(CoordConvNet, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.coord_adder = AddCoordinates(with_r=with_r)
        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_layer(x)

        return x

class ResNextBackbone300(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coordconv = CoordConvNet()
        self.net = timm.create_model('resnext50d_32x4d', in_chans=32, features_only=True, pretrained=True)
        del self.net['layer4']
        del self.net['layer3']

    def forward(self, x):
        x = self.coordconv(x)
        x = self.net(x)
        
        return x[-1] # b x 512 x 16 x 64


class HrnetBackbone300(nn.Module):
    """
    hrnet backbone
    """
    def __init__(self):
        super(HrnetBackbone300,self).__init__()
        self.model = timm.create_model('hrnet_w48',features_only=True, pretrained=True)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv = nn.Conv2d(256,300,kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):
        o = self.model(x)[2]
        o = self.conv(o)

        return o

# =================================================================================
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
        inter_size = num_bn * growth_rate # inter size : 아웃풋을 4 * k로 발표자료애는 있는데 baseline은 3으로 설정
        self.norm1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_size, inter_size, kernel_size=1, stride=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(
            inter_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False # 다시 채널을 growth rate만큼 줄임
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)

# deepCNN300
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
        self.conv = nn.Conv2d( # 채널 수를 반으로 줄이는 작업
            input_size, output_size, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # 크기를 반으로 줄이는 작업

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return self.pool(out)


# deepCNN300
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


# deepCNN300
class DeepCNN300(nn.Module):
    """
    This is specialized to the math formula recognition task
    - three convolutional layers to reduce the visual feature map size and to capture low-level visual features
    (128, 256) -> (8, 32) -> total 256 features
    - transformer layers cannot change the channel size, so it requires a wide feature dimension
    ***** this might be a point to be improved !!
    """

    def __init__(
        self, input_channel, num_in_features, output_channel=256, dropout_rate=0.2, depth=16, growth_rate=24
    ):
        super(DeepCNN300, self).__init__()
        self.conv0 = nn.Conv2d(
            input_channel,  # 3
            num_in_features,  # 48
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # 1/4 (128, 128) -> (32, 32)
        num_features = num_in_features

        self.block1 = DenseBlock(
            num_features,  # 48
            # 항상 denseblock의 아웃풋은 k개로 고정이고 현재 코드에서는 24로 가정
            # 앞에있는 것이 붙고 16번 반복
            growth_rate=growth_rate,  # 48 + growth_rate(24)*depth(16) -> 432
            depth=depth,  # 16?
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2)  # 16 x 16 (dense block 사이의 conv, pooling하는 부분)
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features,  # 128 -> 216아닌가?
            growth_rate=growth_rate,  # 16
            depth=depth,  # 8
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(
            num_features, num_features // 2, kernel_size=1, stride=1, bias=False  # 128
        )

    def forward(self, input):
        out = self.conv0(input)  # (H, V, )
        out = self.relu(self.norm0(out))
        out = self.max_pool(out)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out_before_trans2 = self.trans2_relu(self.trans2_norm(out))
        out_A = self.trans2_conv(out_before_trans2)
        return out_A  # 128 x (16x16)
# =================================================================================

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
    """트랜스 포머의 Attention부분이라고 생각하면 됨"""
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
        ) # transformer feedforward랑 같다고 생각하면 됨
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


class PositionalEncoding2D(nn.Module):
    def __init__(self, in_channels, max_h=64, max_w=128, dropout=0.1):
        super(PositionalEncoding2D, self).__init__()

        # 반을 잘라 concat
        self.h_position_encoder = self.generate_encoder(in_channels // 2, max_h) # max_len, in_channel
        self.w_position_encoder = self.generate_encoder(in_channels // 2, max_w) # max_len, in_channel

        self.h_linear = nn.Linear(in_channels // 2, in_channels // 2)
        self.w_linear = nn.Linear(in_channels // 2, in_channels // 2)

        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)
        i = torch.arange(in_channels).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)
        position_encoder = pos * angle_rates
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])
        return position_encoder  # (Max_len, In_channel)

    def forward(self, input):
        ### Require DEBUG
        b, c, h, w = input.size()
        h_pos_encoding = (
            self.h_position_encoder[:h, :].unsqueeze(1).to(input.get_device())
        )
        h_pos_encoding = self.h_linear(h_pos_encoding)  # [H, 1, D]

        w_pos_encoding = (
            self.w_position_encoder[:w, :].unsqueeze(0).to(input.get_device())
        )
        w_pos_encoding = self.w_linear(w_pos_encoding)  # [1, W, D]

        h_pos_encoding = h_pos_encoding.expand(-1, w, -1)   # h, w, c/2
        w_pos_encoding = w_pos_encoding.expand(h, -1, -1)   # h, w, c/2

        pos_encoding = torch.cat([h_pos_encoding, w_pos_encoding], dim=2)  # [H, W, 2*D]

        pos_encoding = pos_encoding.permute(2, 0, 1)  # [2*D, H, W]

        out = input + pos_encoding.unsqueeze(0)
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
        input_size,
        hidden_dim,
        filter_size,
        head_num,
        layer_num,
        dropout_rate=0.1,
        checkpoint=None,
        backbone = "resnext",
        encoding_type = "baseline"
    ):
        super(TransformerEncoderFor2DFeatures, self).__init__()

        # self.shallow_cnn = DeepCNN300( # denseNet 기반
        #     input_size,
        #     num_in_features=48,
        #     output_channel=hidden_dim,
        #     dropout_rate=dropout_rate,
        # )

        if backbone == "hrnet":
            self.shallow_cnn = HrnetBackbone300() # hrnet사용
        elif backbone == "resnext":
            self.shallow_cnn = ResNextBackbone300() # ResNext사용
        else:
            self.shallow_cnn = DeepCNN300( # denseNet 기반
            input_size,
            num_in_features=48,
            output_channel=hidden_dim,
            dropout_rate=dropout_rate,
            )
        
        if encoding_type  == "baselien":
            self.positional_encoding = PositionalEncoding2D(hidden_dim)
        elif encoding_type == "paper":
            self.positional_encoding = PositionalEncoding2D_paper(hidden_dim)

        # 추가 부분
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
        # 추가 부분
        m = nn.AdaptiveAvgPool2d(1)
        ge = m(out)
        ge = ge.squeeze(-1).squeeze(-1)
        alpha = torch.sigmoid(self.linear_h_2(self.relu(self.linear_h_1(ge))))
        beta = torch.sigmoid(self.linear_h_2(self.relu(self.linear_w_1(ge))))

        out = self.positional_encoding(out, alpha, beta)  # [b, c, h, w]

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
        ) # 인코더에서 넘어온 데이터와 함께 Attention을 해 주는 부분
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.feedforward_layer = Feedforward(
            filter_size=filter_size, hidden_dim=input_size
        )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, tgt, tgt_prev, src, tgt_mask):
        # tgt_prev : 지금 들어간 단어 이전의 단어들에 대해서 q, k, v
        # tgt : 지금 들어간 단어를 포함한 단어들에 대한 q, k, v

        if tgt_prev == None:  # Train (teacher forcing하는 경우  이 케이스로 들어가서 한방에 끝내고 teacher forcing 안하는 경우는 처음 1회 수행때만 수행을 하게 됨)
            att = self.self_attention_layer(tgt, tgt, tgt, tgt_mask)
            out = self.self_attention_norm(att + tgt)

            att = self.attention_layer(out, src, src) # q 는 내꺼, k,v는 encoder에서 넘어온 것(src)
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
    """2D 구조에서 1D로"""
    def __init__(self, in_channels, max_len=500, dropout=0.1):
        super(PositionEncoder1D, self).__init__()

        self.position_encoder = self.generate_encoder(in_channels, max_len) # positional embedding해줌
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
        checkpoint=None,
    ):
        super(TransformerDecoder, self).__init__()

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
        self, src, text, is_train=True, batch_max_length=50, teacher_forcing_ratio=1.0
    ):

        """무슨 소리인지 몰라 다시 복습 필요 """
        if is_train and random.random() < teacher_forcing_ratio: # teacher forcing부분
            tgt = self.text_embedding(text)
            tgt = self.pos_encoder(tgt)
            # 마스크 부분에서 pad 토큰인 것만 마스크 씌움
            tgt_mask = self.pad_mask(text) | self.order_mask(text.size(1)) # order mask : 뒷쪽을 보지 않음, pad_mask : pad 토큰을 고려하지 않음
            for layer in self.attention_layers:
                tgt = layer(tgt, None, src, tgt_mask)
            out = self.generator(tgt)
        else:
            out = []
            num_steps = batch_max_length - 1
            target = torch.LongTensor(src.size(0)).fill_(self.st_id).to(device) # [START] token
            features = [None] * self.layer_num

            for t in range(num_steps):
                target = target.unsqueeze(1)
                tgt = self.text_embedding(target)
                tgt = self.pos_encoder(tgt, point=t)
                tgt_mask = self.order_mask(t + 1) # order mask를 그때그때 생성
                tgt_mask = tgt_mask[:, -1].unsqueeze(1)  # [1, (l+1)] # -1 -> 마지막은 무조건 1 이라서 사실상은 마스킹을 안하는 것이라 생각하면 됨
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

        return out


class SATRN(nn.Module):
    def __init__(self, FLAGS, train_dataset, checkpoint=None):
        super(SATRN, self).__init__()

        self.encoder = TransformerEncoderFor2DFeatures(
            input_size=FLAGS.data.rgb,
            hidden_dim=FLAGS.SATRN.encoder.hidden_dim,
            filter_size=FLAGS.SATRN.encoder.filter_dim,
            head_num=FLAGS.SATRN.encoder.head_num,
            layer_num=FLAGS.SATRN.encoder.layer_num,
            dropout_rate=FLAGS.dropout_rate,
            backbone = FLAGS.SATRN.backbone,
            encoding_type = FLAGS.SATRN.encoding_2d_type
        )

        self.decoder = TransformerDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.SATRN.decoder.src_dim,
            hidden_dim=FLAGS.SATRN.decoder.hidden_dim,
            filter_dim=FLAGS.SATRN.decoder.filter_dim,
            head_num=FLAGS.SATRN.decoder.head_num,
            dropout_rate=FLAGS.dropout_rate,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            layer_num=FLAGS.SATRN.decoder.layer_num,
        )

        self.criterion = (
            nn.CrossEntropyLoss(ignore_index=train_dataset.token_to_id[PAD])
        )  # without ignore_index=train_dataset.token_to_id[PAD]

        if checkpoint:
            self.load_state_dict(checkpoint)

    def forward(self, input, expected, is_train, teacher_forcing_ratio):
        enc_result = self.encoder(input)
        dec_result = self.decoder(
            enc_result,
            expected[:, :-1],
            is_train,
            expected.size(1),
            teacher_forcing_ratio,
        )
        return dec_result