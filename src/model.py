from pathlib import Path
from typing import Dict, Any
import logging

import torch
from collections import OrderedDict
import torch.nn as nn

logger = logging.getLogger(__file__)


class m46(nn.Module):
    def __init__(self, input_shape, model_type='age'):
        super().__init__()

        self.input_shape = input_shape
        self.nchannel = input_shape[0]
        assert model_type in ['age', 'gender']
        self.model_type = model_type

        # Saving args for convinient restoring from ckpt
        self._params = {
            'input_shape': input_shape,
            'model_type': model_type
        }

        self.convolution = nn.Sequential(
            self._vgg_block(self.nchannel, 32, 1),  # Block 1
            self._vgg_block(32, 64, 2),  # Block 2
            self._vgg_block(64, 128, 3),  # Block 3
            self._vgg_block(128, 128, 4),  # Block 4
            self._vgg_block(128, 256, 5),  # Block 5
            self._vgg_block(256, 384, 6),  # Block 6
            nn.Flatten(),
        )

        dummy = torch.randn(1, *input_shape)
        dummy = self.convolution.forward(dummy)
        in_channels = dummy.size()[1:].numel()
        fcc_layers = OrderedDict([
            ('dropout1', nn.Dropout(p=0.3)),
            ('fc1', nn.Linear(in_channels, 2048)),
            ('elu1', nn.ELU(inplace=True)),
            ('dropout2', nn.Dropout(p=0.3)),
            ('fc2', nn.Linear(2048, 2048)),
            ('elu2', nn.ELU(inplace=True)),
            ('preds', nn.Linear(2048, 1)),
        ])
        if self.model_type == 'gender':
            fcc_layers['probs'] = nn.Sigmoid()
        self.fcc = nn.Sequential(fcc_layers)

        if self.model_type == 'age':
            self._loss_function = nn.L1Loss()
        else:
            self._loss_function = nn.BCELoss()
        self._initialize_weights()

    def _vgg_block(self, in_channels, out_channels, block_num, kernel_size=3):
        b = f'block{block_num}_'
        vgg_block = nn.Sequential(OrderedDict([
            (b + 'conv1', nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1)),
            (b + 'elu1', nn.ELU(inplace=True)),
            (b + 'bn1', nn.BatchNorm2d(out_channels)),
            (b + 'conv2', nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1)),
            (b + 'elu2', nn.ELU(inplace=True)),
            (b + 'bn2', nn.BatchNorm2d(out_channels)),
            (b + 'pool', nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=(2, 2)))
        ]))
        return vgg_block

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, labels=None):
        x = self.convolution(x)
        outputs = self.fcc(x)
        return outputs

    def save(self, path_to_save: Path) -> None:
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'checkpoint_data': {'params': self._params}
        }
        torch.save(checkpoint, path_to_save)
        logger.info(f'Model saved to {path_to_save}.')

    @property
    def init_params(self) -> Dict[str, Any]:
        return self._params

    @property
    def loss_function(self) -> nn.modules.Module:
        return self._loss_function

    @classmethod
    def from_ckpt(cls, checkpoint: Path or OrderedDict) -> 'm46':
        ckpt = torch.load(checkpoint, map_location='cpu') if type(checkpoint) == Path else checkpoint
        model = cls(**ckpt['checkpoint_data']['params'])
        model.load_state_dict(ckpt['model_state_dict'])
        if type(checkpoint) == Path:
            logger.info(f'Model was loaded from {checkpoint}.')
        else:
            logger.info(f'Model was loaded from dictionary.')
        return model


def convert_checkpoint(checkpoint: Path or OrderedDict,
                       params: Dict[str, Any]) -> Dict[str, Any]:
    """make use of previous checkpoint format"""

    ckpt = torch.load(checkpoint, map_location='cpu') if isinstance(checkpoint, Path) else checkpoint
    updates = {'.predictions.': '.preds.', '.prediction_probs.': '.probs.'}
    for k in list(ckpt):
        upd = next((u for u in updates if u in k), None)
        if upd:
            ckpt[k.replace(upd, updates[upd])] = ckpt.pop(k)
    checkpoint = {
        'model_state_dict': ckpt,
        'checkpoint_data': {'params': params}
    }
    return checkpoint


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the model
    input_shape = (1, 500, 375)
    model = m46(input_shape, model_type='age').to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)))

    # Print the model
    logger.info(f"Model: {model}")
