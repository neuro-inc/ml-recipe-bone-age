import torch
from collections import OrderedDict
import torch.nn as nn


class m46(nn.Module):
    def __init__(self, input_shape, ngpu, model_type='age'):
        super(m46, self).__init__()
        self.input_shape = input_shape
        self.nchannel = input_shape[0]
        self.ngpu = ngpu
        assert model_type in ['age', 'gender']
        self.model_type = model_type

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
            ('predictions', nn.Linear(2048, 1)),
        ])
        if self.model_type == 'gender':
            fcc_layers['prediction_probs'] = nn.Sigmoid()
        self.fcc = nn.Sequential(fcc_layers)

        if self.model_type == 'age':
            self.loss_function = nn.L1Loss()
        else:
            self.loss_function = nn.BCELoss()
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
        # if labels is not None:
        #     loss = self.loss_function(outputs, labels)
        #     outputs = outputs, loss
        return outputs


if __name__ == '__main__':
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the model
    input_shape = (1, 500, 375)
    model = m46(input_shape, ngpu, model_type='boneage').to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)))

    # Print the model
    print(model)
