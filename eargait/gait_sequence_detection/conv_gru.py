"""ConvGRU model."""

import pytorch_lightning as pl
import torch
from torch import nn


class ConvGRU(pl.LightningModule):
    """ConvGRU model.

    The ConvGRU consists of three convolutional layers and a recurrent unit (GRU).
    """

    def __init__(self, args: dict, input_channels: int, num_classes: int):
        super().__init__()
        # build network with the same options as the BaseCNN + one Gru Layer
        num_conv_layers = args["conv_gru_layers"]
        conv_1_dim = args["conv_gru_1_dim"]
        conv_2_dim = args["conv_gru_2_dim"]
        conv_3_dim = args["conv_gru_3_dim"]
        conv_dim_list = [conv_1_dim, conv_2_dim, conv_3_dim]
        conv_1_kernel = args["conv_gru_1_kernel"]
        conv_2_kernel = args["conv_gru_2_kernel"]
        conv_3_kernel = args["conv_gru_3_kernel"]
        conv_kernel_list = [conv_1_kernel, conv_2_kernel, conv_3_kernel]
        conv_1_dropout = args["conv_gru_1_dropout"]
        conv_2_dropout = args["conv_gru_2_dropout"]
        conv_3_dropout = args["conv_gru_3_dropout"]
        conv_dropout_list = [conv_1_dropout, conv_2_dropout, conv_3_dropout]
        gru_hidden_1 = args["gru_hidden_1"]
        gru_hidden_2 = args["gru_hidden_2"]
        gru_dropout_1 = args["gru_dropout_1"]
        gru_dropout_2 = args["gru_dropout_2"]
        self.gru_layers = args["gru_layers"]
        # setup feature layers
        layers = []
        last_conv_layer_size_features = 1
        for i in range(num_conv_layers):
            layers.append(
                nn.Conv2d(
                    in_channels=last_conv_layer_size_features,
                    out_channels=conv_dim_list[i],
                    kernel_size=(1, conv_kernel_list[i]),
                )
            )
            layers.append(nn.BatchNorm2d(conv_dim_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(conv_dropout_list[i]))
            last_conv_layer_size_features = conv_dim_list[i]
        # output afert conv of N, F, D, S
        # reshape into (F*D) * S
        layers.append(nn.Flatten(start_dim=1, end_dim=2))
        # feature layers as sequential
        self.feature_layers = nn.Sequential(*layers)

        # gru layers, not one stacked layer in order to prune the layers individually later on
        gru_hidden = gru_hidden_1
        self.gru_layer_1 = nn.GRU(
            input_size=input_channels * last_conv_layer_size_features, hidden_size=gru_hidden_1, batch_first=True
        )
        self.gru_dropout_1 = nn.Dropout(gru_dropout_1)
        if self.gru_layers == 2:
            self.gru_layer_2 = nn.GRU(input_size=gru_hidden_1, hidden_size=gru_hidden_2, batch_first=True)
            gru_hidden = gru_hidden_2
        self.gru_dropout_2 = nn.Dropout(gru_dropout_2)
        self.classifier = nn.Linear(gru_hidden, num_classes)

    def forward(self, x):  # noqa
        """Foward processing method of ConGRU."""
        # input of size N, C_in, L_in
        feature_output = self.feature_layers(x)
        # output of size N, C_out, L_out, swap for gru,
        # gru expects shape N, L, H_in
        feature_output = torch.swapaxes(feature_output, 1, 2)
        feature_output, _ = self.gru_layer_1(feature_output)
        feature_output = self.gru_dropout_1(feature_output)
        if self.gru_layers == 2:
            feature_output, _ = self.gru_layer_2(feature_output)
            feature_output = self.gru_dropout_2(feature_output)
        # output of lstm is of shape N, L, H_out
        # only take last step of the sequence for classification
        linear_layer_input = feature_output[:, -1, :]
        out = self.classifier(linear_layer_input)
        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group(ConvGRU)
        # parameters for the Conv LSTM model:
        # default settings equal to best baseCnn
        parser.add_argument("--conv_gru_layers", type=int, default=3, choices=range(1, 4))
        parser.add_argument("--conv_gru_1_dim", type=int, default=70)
        parser.add_argument("--conv_gru_2_dim", type=int, default=50)
        parser.add_argument("--conv_gru_3_dim", type=int, default=30)
        parser.add_argument("--conv_gru_1_kernel", type=int, default=5)
        parser.add_argument("--conv_gru_2_kernel", type=int, default=5)
        parser.add_argument("--conv_gru_3_kernel", type=int, default=3)
        parser.add_argument("--conv_gru_1_dropout", type=float, default=0.5)
        parser.add_argument("--conv_gru_2_dropout", type=float, default=0.3)
        parser.add_argument("--conv_gru_3_dropout", type=float, default=0.2)
        parser.add_argument("--gru_hidden_1", type=int, default=32)
        parser.add_argument("--gru_hidden_2", type=int, default=0)
        parser.add_argument("--gru_dropout_1", type=float, default=0.2)
        parser.add_argument("--gru_dropout_2", type=float, default=0.0)
        parser.add_argument("--gru_layers", type=int, default=1)
        return parent_parser
