"""Deep Learning HAR Predictor.

Wrapper class for the Deep Learning models, that contains the optimizer, the loss function,
and is responsible for logging the whole learning and tests process to tensorboard.
"""

import pytorch_lightning as pl
import torch
from torch import nn

from eargait.gait_sequence_detection.conv_gru import ConvGRU


class HARPredictor(pl.LightningModule):
    """Deep Learning HAR Predictor.

    Wrapper class for the Deep Learning models, that contains the optimizer, the loss function,
    and is responsible for logging the whole learning and tests process to tensorboard.
    """

    def __init__(
        self,
        args: dict,
        input_channels: int,
        num_classes: int,
        weight_decay: float = 0,
        weighted_probabilities: list[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["weighted_probabilities"])
        self.learning_rate = args["learning_rate"]
        self.optimizer = args["optimizer"]
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.model = ConvGRU(args, input_channels, num_classes)
        if weighted_probabilities is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weighted_probabilities))
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, data, labels=None):  # noqa
        output = self.model(data)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
