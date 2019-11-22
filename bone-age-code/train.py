import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import m46
from dataset import split_dataset, BoneAgeDataset, normalize_target
from transforms import get_transform

import pandas as pd
import numpy as np
import copy
from argparse import ArgumentParser
from typing import Dict, Optional, Tuple, List
import re
from pathlib import Path


def train_one_epoch(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    data_loader: DataLoader,
                    device: torch.device) -> float:
    model.train()  # Set model to training mode
    running_loss = 0.0
    dataset_size = 0
    for i, data in enumerate(data_loader, 1):

        # Format batch
        inputs, labels = data['image'], data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        model.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, labels)
        prediction, loss = outputs

        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    print(f'Train loss: {epoch_loss:.4f}')
    return epoch_loss


def evaluate(model: nn.Module,
             data_loader: DataLoader,
             device: torch.device) -> float:
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    dataset_size = 0

    for i, data in enumerate(data_loader, 1):

        # Format batch
        inputs, labels = data['image'], data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs, labels)
            prediction, loss = outputs

        # statistics
        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    print(f'Eval loss: {epoch_loss:.4f}')
    return epoch_loss


def train(model: nn.Module,
          dataloaders: Dict[str, DataLoader],
          save_path: Path,
          n_epoch: int,
          device: torch.device,
          prev_epoch: Optional[int],
          prev_loss: Optional[float],
          ) -> List[Tuple]:

    # Learning rate for optimizers
    # lr = 5e-5
    lr = 2e-5

    # Setup Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    # Lists to keep track of progress
    losses = []
    prev_epoch = 0 if prev_epoch is None else prev_epoch
    best_loss = np.inf if prev_loss is None else prev_loss
    # best_loss = np.inf

    print('Starting Training Loop...')
    for epoch in range(1, n_epoch + 1):
        epoch_stamp = f'Epoch {epoch}/{n_epoch}'
        print(epoch_stamp)
        print('-' * len(epoch_stamp))

        train_loss = train_one_epoch(model, optimizer, dataloaders['train'], device)
        val_loss = evaluate(model, dataloaders['val'], device)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, str(save_path).
                       format(**{'epoch': epoch + prev_epoch, 'epoch_loss': best_loss}))
            print('New accuracy reached. Model saved.')

        losses.append((train_loss, val_loss))
        print()

    print('Done training')
    return losses


def main(data_dir: Path, preprocessing_args: Dict,
         annotation_csv: Path, dataset_split: Tuple,
         prev_ckpt: Optional[Path],
         model_save_dir: Path,
         n_epoch: int, batch_size: int,
         n_workers: int, n_gpu: int
         ) -> None:
    model_save_path = model_save_dir / f'boneage.epoch{{epoch:02d}}-err{{epoch_loss:.3f}}.pth'
    crop_dict = preprocessing_args['crop_dict']
    scale = preprocessing_args['scale']
    input_shape = preprocessing_args['input_shape']

    prev_epoch, prev_loss = None, None
    if prev_ckpt is not None:
        parse = re.match('.+epoch([0-9]+)-err([0-9]+.[0-9]+)', prev_ckpt.stem)
        prev_epoch, prev_loss = int(parse[1]), float(parse[2])

    # Decide which device we want to run on
    device = torch.device('cuda:0' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu')

    # Create the model
    model = m46(input_shape, n_gpu).to(device)
    if prev_ckpt is not None:
        model.load_state_dict(torch.load(prev_ckpt))
        print('Model loaded from', prev_ckpt)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (n_gpu > 1):
        model = nn.DataParallel(model, list(range(n_gpu)))

    # datasets and loaders
    annotation_frame = pd.read_csv(annotation_csv)
    test_fold, n_folds = dataset_split
    train_df, test_df = split_dataset(annotation_frame, test_fold, n_folds, data_dir, gender='a')
    # train_df = train_df.iloc[:160, :]

    data_frames = {'train': train_df, 'val': test_df}
    transforms = {
        phase: get_transform(augmentation=(phase == 'train'), crop_dict=crop_dict, scale=scale)
        for phase in ['train', 'val']
    }
    datasets = {
        phase: BoneAgeDataset(bone_age_frame=data_frames[phase], root=data_dir,
                              transform=transforms[phase], target_transform=normalize_target)
        for phase in ['train', 'val']
    }
    dataloaders = {
        phase: DataLoader(datasets[phase], batch_size=batch_size,
                          shuffle=(phase == 'train'), num_workers=n_workers)
        for phase in ['train', 'val']
    }
    train(model=model,
          dataloaders=dataloaders,
          save_path=model_save_path,
          n_epoch=n_epoch,
          device=device,
          prev_epoch=prev_epoch, prev_loss=prev_loss)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path,
                        default=Path(__file__).absolute().parent.parent / 'data/train')
    parser.add_argument('--crop_center', type=Tuple[int, ...], default=(1040, 800))
    parser.add_argument('--crop_size', type=Tuple[int, ...], default=(2000, 1500))
    parser.add_argument('--scale', type=float, default=0.25)
    parser.add_argument('--annotation_csv', type=Path,
                        default=Path(__file__).absolute().parent.parent / 'data/train.csv')
    parser.add_argument('--test_fold', type=int, default=10)
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--prev_ckpt', type=Path, default=None)
    parser.add_argument('--model_save_dir', type=Path,
                        default=Path(__file__).absolute().parent.parent / 'models')
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=4)
    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    data_dir = args.data_dir
    preprocess_args = {
        'crop_dict': {'crop_center': args.crop_center, 'crop_size': args.crop_size},
        'scale': args.scale,
    }
    h, w = preprocess_args['crop_dict']['crop_size']
    input_shape = {'input_shape': (1, int(h * args.scale), int(w * args.scale))}
    preprocess_args.update(input_shape)
    annotation_csv = args.annotation_csv
    dataset_split = (args.test_fold, args.n_folds)
    prev_ckpt = args.prev_ckpt
    model_save_dir = args.model_save_dir
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    n_gpu = args.n_gpu
    n_workers = args.n_workers

    print('Input shape', 'x'.join(map(str, input_shape['input_shape'])), '[CxHxW]')

    main(
        data_dir=data_dir,
        preprocessing_args=preprocess_args,
        annotation_csv=annotation_csv, dataset_split=dataset_split,
        prev_ckpt=prev_ckpt,
        model_save_dir=model_save_dir,
        n_epoch=n_epoch,
        batch_size=batch_size,
        n_gpu=n_gpu,
        n_workers=n_workers,
    )
