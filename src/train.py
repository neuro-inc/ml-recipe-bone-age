import warnings
from argparse import ArgumentParser, Namespace
warnings.simplefilter('ignore')

from collections import OrderedDict
from typing import Tuple
from pathlib import Path

import torch
from torch import device as tdevice
from torch.cuda import is_available

import catalyst.dl.callbacks as clb
from catalyst.dl.runner import SupervisedRunner
from catalyst.utils import set_global_seed

from model import m46
from dataset import get_loaders
from const import LOG_DIR, DATA_PATH, MODELS_DIR


def main(args: Namespace) -> None:
    input_shape = (1, int(args.crop_size[0] * args.scale), int(args.crop_size[1] * args.scale))
    print('Input shape', 'x'.join(map(str, input_shape)), '[CxHxW]')

    set_global_seed(args.seed)

    train_loader, test_loader = get_loaders(args)
    loaders = OrderedDict([('train', train_loader), ('valid', test_loader)])

    model = m46(input_shape=input_shape, model_type=args.model_type)
    criterion = model.loss_function
    optimizer = torch.optim.Adam(lr=2e-5, betas=(0.5, 0.999), params=model.parameters())

    output_key = 'probs' if args.model_type == 'gender' else 'preds'
    runner = SupervisedRunner(input_key='image', output_key=output_key,
                              input_target_key='label',
                              device=args.device if is_available() else tdevice('cpu')
                              )
    callbacks = [clb.CriterionCallback(input_key='label', output_key=output_key)]
    if args.model_type == 'gender':
        callbacks += [clb.AccuracyCallback(prefix='accuracy', input_key='label',
                                           output_key=output_key, accuracy_args=[1],
                                           threshold=.5, num_classes=1, activation="none")]

    runner.train(
        model=model, criterion=criterion, optimizer=optimizer,
        scheduler=None, loaders=loaders, logdir=str(args.logdir),
        num_epochs=args.n_epoch, verbose=True, main_metric='loss',
        valid_loader='valid', callbacks=callbacks, minimize_metric=True,
        checkpoint_data={'params': model.init_params}
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default=DATA_PATH / 'train')
    parser.add_argument('--crop_center', type=Tuple[int], default=(1040, 800))
    parser.add_argument('--crop_size', type=Tuple[int], default=(2000, 1500))
    parser.add_argument('--scale', type=float, default=0.25)
    parser.add_argument('--annotation_csv', type=Path, default=DATA_PATH / 'train.csv')
    parser.add_argument('--dataset_split', type=Tuple[int], default=(10, 10))   # (test_fold, n_folds)
    parser.add_argument('--model_type', type=str, default='age')
    parser.add_argument('--prev_ckpt', type=Path, default=None)
    parser.add_argument('--model_save_dir', type=Path, default=MODELS_DIR)
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=tdevice, default='cuda:0')
    parser.add_argument('--logdir', type=Path, default=LOG_DIR)
    return parser


if __name__ == '__main__':
    main(args=get_parser().parse_args())
