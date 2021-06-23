from pathlib import Path
import torch
import pandas as pd
import cv2
import numpy as np
import logging
from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt
from transforms import get_transform
from itertools import islice
from typing import Tuple, Dict
from argparse import Namespace
from torch.utils.data import DataLoader

from src.const import DATA_PATH


logger = logging.getLogger()


def split_dataset(df, test_fold, nfolds, root_dir, gender='a'):
    """
    Split dataframe into train/test, where test is `test_fold` and train is remaining folds
    Args:
        df (DataFrame): pandas DataFrame with annotations.
        nfolds (int): number of folds
        test_fold (int): test fold [1...nfolds]
        root_dir (string or Path): directory with radiographs
        gender ('m'|'f'|'a'): filter dataset based on gender, [m]ale, [f]emale, gender [a]gnostic
    """
    assert gender in ['a', 'm', 'f']
    assert 1 <= test_fold <= nfolds
    root_dir = Path(root_dir)

    # make sure all listed radiographs are actually present
    radiograph_set = {int(f.stem) for f in root_dir.glob('*.png')}
    df['id'] = df['id'].astype(int)
    df = df.loc[df['id'].isin(radiograph_set)]

    if gender == 'm':
        df = df.loc[df['male']]
    elif gender == 'f':
        df = df.loc[~df['male']]
    df_len = len(df)

    fold_length = int(np.ceil(df_len / nfolds))
    test_index = range(fold_length * (test_fold - 1), min(df_len, fold_length * test_fold))
    train_index = [i for i in range(df_len) if i not in test_index]
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    assert pd.concat([train_df, test_df]).loc[df.index].equals(df)

    return train_df, test_df


def normalize_target(x, reverse_norm=False):
    if reverse_norm:
        return x * 120 + 120
    else:
        return (x - 120) / 120


class BoneAgeDataset(VisionDataset):
    """Bone Age dataset."""

    def __init__(self, bone_age_frame, root, transform=None, target_transform=None, model_type='age'):
        """
        Args:
            bone_age_frame (DataFrame): pandas DataFrame with annotations.
            root (string or Path): directory with all the images.
            transform (callable, optional): optional transform to be applied
                on a sample.
            model_type (string): target to predict, can be either boneage or gender
        """
        super().__init__(root=Path(root), transform=transform,
                                             target_transform=target_transform)
        bone_age_frame['id'] = bone_age_frame['id'].astype(int)
        self.bone_age_frame = bone_age_frame#.loc[bone_age_frame['id']]

        assert model_type in ['age', 'gender']
        self.model_type = model_type

    def __len__(self):
        return len(self.bone_age_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_id = self.bone_age_frame['id'].iloc[idx].astype(str)
        img_name = self.root / (img_id + '.png')
        image = cv2.imread(str(img_name), flags=cv2.IMREAD_GRAYSCALE)
        if (self.model_type == 'age') and ('boneage' in self.bone_age_frame.columns):
            target = self.bone_age_frame['boneage'].iloc[idx]
            target = np.array(target).astype(np.float32)
        elif (self.model_type == 'gender') and ('male' in self.bone_age_frame.columns):
            target = self.bone_age_frame['male'].iloc[idx] * 1.0
            target = np.array(target).astype(np.float32)
        else:
            target = None
        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = {'image': image, 'label': target, 'id': img_id}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def get_loaders(args: Namespace)-> Tuple[DataLoader, DataLoader]:
    crop_args = {'crop_center': args.crop_center,
                 'crop_size': args.crop_size}
    annotation_frame = pd.read_csv(args.annotation_csv)
    test_fold, n_folds = args.dataset_split
    logger.info(f"Input dataset split: {test_fold}, {n_folds}")
    train_df, test_df = split_dataset(annotation_frame, test_fold, n_folds, args.data_dir, gender='a')
    logger.info(f"Input dataset split DF: {len(train_df)}, {len(test_df)}")
    # train_df = train_df.iloc[:160, :]

    data_frames = {'train': train_df, 'val': test_df}
    transforms = {
        phase: get_transform(augmentation=(phase == 'train'), crop_dict=crop_args, scale=args.scale)
        for phase in ['train', 'val']
    }
    datasets = {
        phase: BoneAgeDataset(bone_age_frame=data_frames[phase], root=args.data_dir,
                              transform=transforms[phase],
                              target_transform=None if args.model_type == 'gender' else normalize_target,
                              model_type=args.model_type)
        for phase in ['train', 'val']
    }
    dataloaders = [
        DataLoader(datasets[phase], batch_size=args.batch_size, shuffle=(phase == 'train'),
                   num_workers=args.n_workers)
        for phase in ['train', 'val']
    ]
    return dataloaders


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    model_type = 'age'
    bone_age_frame = pd.read_csv(DATA_PATH / 'train.csv')
    root_dir = DATA_PATH / 'train'

    nfolds = 9
    for test_fold in range(1, nfolds + 1):
        train_df, test_df = split_dataset(bone_age_frame, test_fold, nfolds, '../data/train', gender='a')
        logger.info(f"train_df: {len(train_df)}, test_df: {len(test_df)}")

    crop_dict = {'crop_center': (1040, 800), 'crop_size': (2000, 1500)}
    scale = 0.25
    train_transform = get_transform(augmentation=False, crop_dict=crop_dict, scale=scale)

    train_df, _ = split_dataset(bone_age_frame, 5, 5, root_dir, gender='a')
    boneage_dataset = BoneAgeDataset(bone_age_frame=train_df, root=root_dir,
                                     transform=train_transform,
                                     target_transform=None if model_type == 'gender' else normalize_target,
                                     model_type=model_type)
    nimages = 4
    fig = plt.figure(figsize=(18, 8))
    for i, sample in enumerate(islice(boneage_dataset, nimages), 1):
        image, label, img_id = sample['image'], sample['label'], sample['id']
        if torch.is_tensor(image):
            image = np.squeeze(image.numpy())
            label = label.item()
        if model_type == 'age':
            label = normalize_target(label, reverse_norm=True)
        elif model_type == 'gender':
            label = ['female', 'male'][int(label)]
        ax = plt.subplot(1, nimages, i)
        size_stamp = 'x'.join(map(str, image.shape))
        if model_type == 'age':
            title = f'id {img_id}, {label:n} months\nh, w={size_stamp}'
        else:
            title = f'id {img_id}, {label}\nh, w={size_stamp}'
        ax.set_title(title, fontsize=18)
        ax.axis('off')
        ax.imshow(image, cmap='Greys_r')
    plt.tight_layout()
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()
