import timm
import torch
from prettyprinter import pprint

pprint(timm.list_models(pretrained=True))
# resnext50d_32x4d
m = timm.create_model('swsl_resnext101_32x16d', features_only=True, in_chans=1, pretrained=True)
print(f'Feature channels: {m.feature_info.channels()}')
# del m['final_conv']
# del m['stages_2']
# del m['stages_3']
o = m(torch.randn(2, 1, 128, 512))
pprint([x for x in m])
for x in o:
  print(x.shape)