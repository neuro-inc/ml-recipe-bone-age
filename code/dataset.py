import sys
sys.path.append("..")
from pathlib import Path
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from code.transforms import get_transform
from itertools import islice


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
    radiograph_list = [f.stem for f in root_dir.glob('*.png')]
    df = df.loc[df['id'].isin(radiograph_list)]

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


class BoneAgeDataset(Dataset):
    """Bone Age dataset."""

    def __init__(self, bone_age_frame, root_dir, transform=None):
        """
        Args:
            bone_age_frame (DataFrame): pandas DataFrame with annotations.
            root_dir (string or Path): directory with all the images.
            transform (callable, optional): optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        # make sure all listed radiographs are actually present
        radiographs = [f.stem for f in self.root_dir.glob('*.png')]
        self.bone_age_frame = bone_age_frame.loc[bone_age_frame['id'].isin(radiographs)]
        self.transform = transform

    def __len__(self):
        return len(self.bone_age_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_id = self.bone_age_frame['id'].iloc[idx].astype(str)
        img_name = self.root_dir / (img_id + '.png')
        image = cv2.imread(str(img_name), flags=cv2.IMREAD_GRAYSCALE)
        if 'boneage' in self.bone_age_frame.columns:
            label = self.bone_age_frame['boneage'].iloc[idx].astype('float')
            label = (label - 120) / 120
        else:
            label = None
        sample = {'image': image, 'label': label, 'id': img_id}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    df = pd.read_csv('f:/Sandbox/hand/data/train.csv')
    nfolds = 9
    for test_fold in range(1, nfolds + 1):
        train_df, test_df = split_dataset(df, test_fold, nfolds, 'f:/Sandbox/hand/data/fitted_train', gender='m')
        print(len(train_df), len(test_df))

    crop_dict = {'center': (1040, 800), 'crop_size': (2000, 1500)}
    scale = 0.25
    train_transform = get_transform(augmentation=False, crop_dict=crop_dict, scale=scale)

    bone_age_frame = pd.read_csv('f:/Sandbox/hand/data/train.csv')
    root_dir = 'f:/Sandbox/hand/data/fitted_train'
    train_df, _ = split_dataset(bone_age_frame, 5, 5, root_dir, gender='a')
    boneage_dataset = BoneAgeDataset(bone_age_frame=train_df, root_dir=root_dir, transform=train_transform)

    nimages = 4
    fig = plt.figure(figsize=(18, 8))
    for i, sample in enumerate(islice(boneage_dataset, nimages), 1):
        image, label, img_id = sample['image'], sample['label'], sample['id']
        if torch.is_tensor(image):
            image = np.squeeze(image.numpy())
            label = label.item()

        ax = plt.subplot(1, nimages, i)
        plt.tight_layout()
        ax.set_title(f'id {img_id}, {label:n} months\nh, w={image.shape}', fontsize=18)
        ax.axis('off')
        ax.imshow(image, cmap='Greys_r')
    plt.tight_layout()
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()
