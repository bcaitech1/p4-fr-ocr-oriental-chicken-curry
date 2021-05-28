import torch
import torch.nn as nn

# from code.networks.modules.tps_transformers import TPS_SpatialTransformerNetwork
# from code.networks.modules.srn_modules import Transforme_Encoder, SRN_Decoder, Torch_transformer_encoder
# from code.networks.modules.feature_extractor import ResNet
from moduels.tps_transformers import TPS_SpatialTransformerNetwork
from moduels.srn_modules import Transforme_Encoder, SRN_Decoder, Torch_transformer_encoder
from moduels.feature_extractor import ResNet



class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
                F=config.model.num_fiducial, 
                I_size=(config.inH, config.inW), I_r_size=(config.inH, config.inW),
                I_channel_num=config.model.inp_channel)

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet(config.model.inp_channel, config.model.out_channel)
        self.FeatureExtraction_output = config.model.out_channel # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = Transforme_Encoder(n_layers=3, n_position=config.model.position_dim)
        self.SequenceModeling_output = 512

        self.Prediction = SRN_Decoder(n_position=config.model.position_dim, 
                                      N_max_character=config.model.max_sequence+1,
                                      n_class=config.model.token_len)


    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        # if self.stages['Feat'] == 'AsterRes' or self.stages['Feat'] == 'ResnetFpn':
        #     b, c, h, w = visual_feature.shape
        #     visual_feature = visual_feature.permute(0, 1, 3, 2)
        #     visual_feature = visual_feature.contiguous().view(b, c, -1)
        #     visual_feature = visual_feature.permute(0, 2, 1)  # batch, seq, feature
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature, src_mask=None)[0]

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature)
        return prediction


if __name__ == "__main__":
    from code.dataset import dataset_loader
    from code.flags import Flags
    from torchinfo import summary
    from torchvision.transforms import transforms
    import yaml

    START = "<SOS>"
    END = "<EOS>"
    PAD = "<PAD>"
    SPECIAL_TOKENS = [START, END, PAD]

    def load_vocab(tokens_paths):
        tokens = []

        for tokens_file in tokens_paths:
            with open(tokens_file, "r") as fd:
                reader = fd.read()
                for token in reader.split("\n"):
                    if token not in tokens:
                        tokens.append(token)
        tokens.extend(SPECIAL_TOKENS)
        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        id_to_token = {i: tok for i, tok in enumerate(tokens)}
        return token_to_id, id_to_token

    with open("/opt/ml/code/configs/SRN.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key, val in config.items():
        print(key, ":", val)

    token_to_id, id_to_token = load_vocab(["/opt/ml/input/data/train_dataset/tokens.txt"])
    tfms = transforms.Compose(
        [
            # Resize so all images have the same size
            transforms.Resize((config['inH'], config['inW'])),
            transforms.ToTensor(),
        ]
    )

    flags = Flags("/opt/ml/code/configs/SRN.yaml").get()
    train_data_loader, validation_data_loader, train_dataset, valid_dataset = dataset_loader(flags, tfms)
    print(train_data_loader)


    #
    # m = Model(config)
    # summary(m)
    #
    # m = m.cuda()
    # inp = torch.rand((2, 1, 128, 256)).cuda()
    # out = m(inp, None, True)
    # for o in out:
    #     print(o.shape)