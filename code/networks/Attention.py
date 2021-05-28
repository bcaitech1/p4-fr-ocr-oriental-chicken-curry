import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

from data.dataset import START, PAD


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
                               0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
                               1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
                               2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x

class CNN(nn.Module):
    def __init__(self, nc, leakyRelu=False):
        super(CNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2] # kernel size (3*3conv, 3*3conv...)
        ps = [1, 1, 1, 1, 1, 1, 0] # padding 
        ss = [1, 1, 1, 1, 1, 1, 1] # stride
        nm = [64, 128, 256, 256, 512, 512, 512] # channel수(마지막 채널이 512가 됨)

        def convRelu(i, batchNormalization=False):
            cnn = nn.Sequential()
            nIn = nc if i == 0 else nm[i - 1] # channel in (첫번째 채널 수는 인자로 받아옴)
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True)) # leakyRelu사용
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True)) # relu사용
            return cnn

        # convolution6번, pooling4번
        self.conv0 = convRelu(0)
        self.pooling0 = nn.MaxPool2d(2, 2)
        self.conv1 = convRelu(1)
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.conv2 = convRelu(2, True)
        self.conv3 = convRelu(3)
        self.pooling3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1)) # h 부분의 길이가 늘어남
        self.conv4 = convRelu(4, True)
        self.conv5 = convRelu(5)
        self.pooling5 = nn.MaxPool2d((2, 2), (2, 1), (0, 1)) # h 부분의 길이가 늘어남
        self.conv6 = convRelu(6, True)
    
    def forward(self, input):
        out = self.conv0(input)     # [batch size, 64, 128, 128]
        out = self.pooling0(out)    # [batch size, 64, 64, 64]
        out = self.conv1(out)       # [batch size, 128, 64, 64]
        out = self.pooling1(out)    # [batch size, 128, 32, 32]
        out = self.conv2(out)       # [batch size, 256, 32, 32]
        out = self.conv3(out)       # [batch size, 256, 32, 32]
        out = self.pooling3(out)    # [batch size, 256, 16, 33]
        out = self.conv4(out)       # [batch size, 512, 16, 33]
        out = self.conv5(out)       # [batch size, 512, 16, 33]
        out = self.pooling5(out)    # [batch size, 512, 8, 34]
        out = self.conv6(out)       # [batch size, 512, 7, 33] # b, c, h, w
        return out

class AttentionCell(nn.Module):
    def __init__(self, src_dim, hidden_dim, embedding_dim, num_layers=1, cell_type='LSTM'):
        super(AttentionCell, self).__init__()
        self.num_layers = num_layers

        # 어떠한 작업을 하려고?
        """
        원래는 - >Encoder에서 나온 Key 벡터와 Decoder에서 나온 Query 벡터를 내적하여 Attention score를 만든 다음 
        softmax취해서 나온 값을 다시 cnn 아웃풋(value)에 곱하고 weighted sum을 취해줌

        다른방법 -> Decoder에서 나온 값을 linear를 통과 시켜(h2h), Encoder에서 나온 값도 linera를 통과시킴(i2h)
        """
        self.i2h = nn.Linear(src_dim, hidden_dim, bias=False)
        self.h2h = nn.Linear(
            hidden_dim, hidden_dim
        )  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_dim, 1, bias=False)

        # cell type에 따라서 만들어줌
        if num_layers == 1:
            if cell_type == 'LSTM':
                self.rnn = nn.LSTMCell(src_dim + embedding_dim, hidden_dim)
            elif cell_type == 'GRU':
                self.rnn = nn.GRUCell(src_dim + embedding_dim, hidden_dim)
            else:
                raise NotImplementedError
        else: # layer가 1개 이상일 때
            if cell_type == 'LSTM':
                self.rnn = nn.ModuleList(
                    [nn.LSTMCell(src_dim + embedding_dim, hidden_dim)]
                    + [
                        nn.LSTMCell(hidden_dim, hidden_dim)
                        for _ in range(num_layers - 1)
                    ]
                )
            elif cell_type == 'GRU':
                self.rnn = nn.ModuleList(
                    [nn.GRUCell(src_dim + embedding_dim, hidden_dim)]
                    + [
                        nn.GRUCell(hidden_dim, hidden_dim)
                        for _ in range(num_layers - 1)
                    ]
                )
            else:
                raise NotImplementedError

        self.hidden_dim = hidden_dim

    def forward(self, prev_hidden, src, tgt):   # src: [b, L, c]
        src_features = self.i2h(src)  # [b, L, h] source feature추출
        if self.num_layers == 1:
            prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)    # [b, 1, h] 0-> hidden state, 1-> cell state
        else:
            prev_hidden_proj = self.h2h(prev_hidden[-1][0]).unsqueeze(1)    # [b, 1, h]

        # hidden 에다가 Attention을 더한 다음 score층을 통과
        attention_logit = self.score(
            torch.tanh(src_features + prev_hidden_proj) # [b, L, h]
        )  # [b, L, 1]

        alpha = F.softmax(attention_logit, dim=1)  # [b, L, 1] softmax취해줌
        context = torch.bmm(alpha.permute(0, 2, 1), src).squeeze(1)  # [b, c] alpha값을 value에 곱하여 context output추출(bmm이 그 역활을 함)

        concat_context = torch.cat([context, tgt], 1)  # [b, c+e]  기존 context에다가 concatenate시켜줌

        if self.num_layers == 1: # 층이 한개일 때
            cur_hidden = self.rnn(concat_context, prev_hidden)
        else: # 층이 여러개일 때
            cur_hidden = []
            for i, layer in enumerate(self.rnn):
                if i == 0:
                    concat_context = layer(concat_context, prev_hidden[i])
                else:
                    concat_context = layer(concat_context[0], prev_hidden[i])
                cur_hidden.append(concat_context)

        return cur_hidden, alpha


class AttentionDecoder(nn.Module):
    """Attention의 디코더 부분
    """
    def __init__(
        self,
        num_classes,
        src_dim,
        embedding_dim,
        hidden_dim,
        pad_id,
        st_id,
        num_layers=1,
        cell_type='LSTM',
        checkpoint=None,
    ):
        super(AttentionDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes + 1, embedding_dim) # 다음 character들이 lstm에 들어가기 전에 Embedding을 해 줌
        self.attention_cell = AttentionCell(
            src_dim, hidden_dim, embedding_dim, num_layers, cell_type
        ) # 실제로 Attention Cell이 Attention을 계산
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.generator = nn.Linear(hidden_dim, num_classes) # hidden dimension을 통해 클래스 예측
        self.pad_id = pad_id
        self.st_id = st_id

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(
        self, src, text, teacher_forcing_ratio, epoch, ratio_cycle,is_train=True, batch_max_length=50
    ):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [START] token. text[:, 0] = [START].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = src.size(0)
        num_steps = batch_max_length - 1  # +1 for [s] at end of sentence.

        output_hiddens = (
            torch.FloatTensor(batch_size, num_steps, self.hidden_dim)
            .fill_(0)
            .to(device)
        )# lstm의 outpu으로 나온 hidden state들을 다 모아놓음(num_steps-> lstm의 길이)
        # 모아두웠다가 matrixs로 만들어 한번에 linear층을 태워 class 예측

        if self.num_layers == 1:
            hidden = (
                torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
            ) # 두개를 만들어 놓은 이유는 lstm은 hidden state, cell state 두개의 state들이 존재하기 때문 
        else:
            hidden = [
                (
                    torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                    torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                )
                for _ in range(self.num_layers) # layer가 한개 이상일 경우
            ]
        
        # teacher forcing 값 지정
        if is_train:
            tf_value = teacher_forcing_ratio[epoch // ratio_cycle]
        else:
            tf_value = 0

        # teacher forcing을 하냐 안하냐에 따라 달라지는 부분
        if is_train and random.random() < tf_value:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                embedd = self.embedding(text[:, i]) # 정답에서 i번째 것을 들고 와서 embedding layer를 통과시킴
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, src, embedd) # embdding된 것과 CNN에서 온 것과 hidden state값을 attemtion cell에 넣음 -> 다시 hidden과 attention score alpha값을 받아옴
                if self.num_layers == 1:
                    output_hiddens[:, i, :] = hidden[
                        0
                    ]  # LSTM hidden index (0: hidden, 1: Cell)
                else:
                    output_hiddens[:, i, :] = hidden[-1][0] # layer층이 하나가 아니라면 최상단에 있는 층에 있는 것을 가져옴
            probs = self.generator(output_hiddens)

        else:  # teacher forcing을 하지 않았을 떄
            targets = (
                torch.LongTensor(batch_size).fill_(self.st_id).to(device)
            )  # [START] token
            probs = (
                torch.FloatTensor(batch_size, num_steps, self.num_classes)
                .fill_(0)
                .to(device)
            )

            for i in range(num_steps):
                embedd = self.embedding(targets) # start토큰을 넣고 임베딩
                hidden, alpha = self.attention_cell(hidden, src, embedd) # attention cell에 넣어줌
                if self.num_layers == 1:
                    probs_step = self.generator(hidden[0]) # 여기서 차이인데 아웃풋이 나올때 마다 linear layer통과(모아서 하는 것이 아닌) -> 이번에 나온 결과를 뽑아야 다음번 input으로 넣을 수 있기 때문
                else:
                    probs_step = self.generator(hidden[-1][0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input # 다음 타겟이 나오고 다음번 embedding을 거쳐 input으로 들어가게 됨 

        return probs  # batch_size x num_steps x num_classes


class BidirectionalLSTM(nn.Module):
    """bidirectional LSTM
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class Attention(nn.Module):
    """실제 모델의 클래스
    """
    def __init__(
        self,
        FLAGS,
        train_dataset,
        checkpoint=None,
    ):
        super(Attention, self).__init__()
        
        """Feature Extraction"""
        if FLAGS.Attention.encoder_type == 'cnn':
            self.encoder = CNN(FLAGS.data.rgb)
        elif FLAGS.Attention.encoder_type == 'resnet':
            self.encoder = ResNet_FeatureExtractor(FLAGS.data.rgb)
        
        self.seq = nn.Sequential(
                BidirectionalLSTM(FLAGS.Attention.src_dim, FLAGS.Attention.seq_hidden_dim, FLAGS.Attention.seq_hidden_dim),
                BidirectionalLSTM(FLAGS.Attention.seq_hidden_dim, FLAGS.Attention.seq_hidden_dim, FLAGS.Attention.src_dim))
        
        self.decoder = AttentionDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.Attention.src_dim, # CNN 에서 넘어오는게 src라고 생각하면 됨(config에서 정보 가져옴)
            embedding_dim=FLAGS.Attention.embedding_dim, # config에서 정보 가져옴
            hidden_dim=FLAGS.Attention.hidden_dim, # config에서 정보 가져옴
            pad_id=train_dataset.token_to_id[PAD], # padding token id
            st_id=train_dataset.token_to_id[START], # start token id
            num_layers=FLAGS.Attention.layer_num, # layer개수
            cell_type=FLAGS.Attention.cell_type) # cell type

        self.criterion = ( # 모델 안 loss function
            nn.CrossEntropyLoss()
        )

        if checkpoint:
            self.load_state_dict(checkpoint)
    
    def forward(self, input, expected, is_train, teacher_forcing_ratio, epoch, ratio_cycle):
        """
        
        Args:
            input : 이미지
            expected : 기대값
            is_train : train mode의 유무
            treacher_forcing_ratio(list) : teacher forcing ratio
            epoch(int) : 현재 진행 epoch
            ratio_cycle(int) : teacher forcing 주기
            is_train(boolean) : True - 훈련모드, False - validation모드
        """
        """Feature Extraction"""
        out = self.encoder(input)
        b, c, h, w = out.size()
        out = out.view(b, c, h * w).transpose(1, 2)  # [b, h x w, c] -> CNN feature를 attention으로 사용하게 됨

        # """sequence process"""
        # out = self.seq(out)

        """predeiction"""
        output = self.decoder(
            src = out, 
            text = expected, 
            is_train = is_train, 
            teacher_forcing_ratio = teacher_forcing_ratio,
            epoch = epoch,
            ratio_cycle = ratio_cycle,
            batch_max_length=expected.size(1),
        )      # [b, sequence length, class size], 디코더에 해당 Attention 을 넣어주게 됨
        return output
