import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
  """1x1 convolution"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_sinusoid_encoding(n_position, feat_dim, wave_length=10000):
  # [n_position]
  positions = torch.arange(0, n_position)#.cuda()
  # [feat_dim]
  dim_range = torch.arange(0, feat_dim)#.cuda()
  dim_range = torch.pow(wave_length, 2 * (dim_range // 2) / feat_dim)
  # [n_position, feat_dim]
  angles = positions.unsqueeze(1) / dim_range.unsqueeze(0)
  angles = angles.float()
  angles[:, 0::2] = torch.sin(angles[:, 0::2])
  angles[:, 1::2] = torch.cos(angles[:, 1::2])
  return angles


class AsterBlock(nn.Module):

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(AsterBlock, self).__init__()
    self.conv1 = conv1x1(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

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


class ResNet_ASTER(nn.Module):
  """For aster or crnn
     borrowed from: https://github.com/ayumiymk/aster.pytorch
  """
  def __init__(self, in_channels=1, out_channel=300, n_group=1):
    super(ResNet_ASTER, self).__init__()
    self.n_group = n_group

    in_channels = in_channels
    self.layer0 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True))

    self.inplanes = 32
    self.layer1 = self._make_layer(32, 32,  3, [2, 2]) # [16, 50]
    self.layer2 = self._make_layer(32, 64,  4, [2, 2]) # [8, 25]
    self.layer3 = self._make_layer(64, 128, 6, [2, 2]) # [4, 25]
    self.layer4 = self._make_layer(128, 256, 6, [1, ]) # [2, 25]
    self.layer5 = self._make_layer(256, 256, 3, [1, 1]) # [1, 25]
    self.layer5_1 = self._make_layer(256, 256, 3, [1, 1])

    self.gap = nn.AdaptiveAvgPool2d(1)
    self.linear1 = nn.Linear(256, 512)
    self.linear2 = nn.Linear(512, 1024)
    self.linear3 = nn.Linear(1024, 1)
    self.sigmoid = nn.Sigmoid()

    self.last_layer = self._make_layer(512, 300, 3, [1, 1])

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, inplanes, planes, blocks, stride):
    downsample = None
    if stride != [1, 1] or inplanes != planes:
      downsample = nn.Sequential(
          conv1x1(inplanes, planes, stride),
          nn.BatchNorm2d(planes))

    layers = []
    layers.append(AsterBlock(inplanes, planes, stride, downsample))
    inplanes = planes
    for _ in range(1, blocks):
      layers.append(AsterBlock(inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):

    x0 = self.layer0(x)
    x1 = self.layer1(x0)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)    
    x5 = self.layer5(x4)
    x5_1 = self.layer5(x4)

    out = self.last_layer(torch.cat((x5, x5_1), dim=1))
    
    x_l1 = self.linear1(self.gap(x5_1).view(-1, 256))    
    x_l2 = self.linear2(x_l1)
    x_l3 = self.linear3(x_l2)
    aux_out = self.sigmoid(x_l3.view(x_l3.size(0)))

    return out, aux_out



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
            growth_rate=growth_rate,  # 48 + growth_rate(24)*depth(16) -> 432
            depth=depth,  # 16?
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2)  # 16 x 16
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features,  # 128
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