import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image
import random
from torchvision import transforms


# Generic utilities
# =================

def detect_classes(labels):
    assert len(set(labels)) == len(labels)
    classes = labels
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

shuffle = lambda coll: random.sample(coll, len(coll))

class RandomImagePicker(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        mapping = {}
        for i,group in enumerate(dataset.group):
            mapping.setdefault(group, []).append(i)
        self.group = shuffle(list(mapping.keys()))
        self.lists = [shuffle(mapping[g]) for g in self.group]
        self.y = pd.Series(self.dataset.y.iloc[[lst[0] for lst in self.lists]].values, self.group)
        print("RandomImagePicker: %d images -> %d groups" % (len(self.dataset), len(self)))

    def __getitem__(self, index):
        lst = self.lists[index]
        return self.dataset[random.choice(lst)]

    def __len__(self):
        return len(self.lists)

    @property
    def classes(self):
        return self.dataset.classes


# Challenge-specific utilities
# ============================

class TensorDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, min_key=None, max_key=None, key_modulus=10**6, classes=None):
        self.root = root
        S = 256
        C = 3
        self.y = pd.Series.from_csv('working/dataset/%s_y.csv' % root)
        try:
            self.group = pd.Series.from_csv('working/dataset/%s_group.csv' % root)
        except:
            self.group = None
        min_i = np.flatnonzero(self.y.index.values%key_modulus >= min_key)[0] if min_key is not None else None
        max_i = np.flatnonzero(self.y.index.values%key_modulus >= max_key)[0] if max_key is not None else None
        self.y = self.y.iloc[min_i:max_i]
        if self.group is not None:
            self.group = self.group.iloc[min_i:max_i]
        print("TensorDataset size: %d" % len(self.y))
        self.X = np.memmap('working/dataset/%s_X.u8' % (root,), dtype=(np.uint8, (S, S, C)), mode='r')[min_i:max_i]
        self.indices = self.y.index
        self.set_class_list([np.nan] if self.y.dtype==np.float64 else sorted(set(self.y.values)))
        self.transform = transform
        self.target_transform = target_transform
        assert len(self.X) == len(self.y)

    def set_class_list(self, classes):
        self.classes, self.class_to_idx = detect_classes(classes)

    def __getitem__(self, index):
        img = self.X[index]
        img = Image.fromarray(img)
        target = self.class_to_idx.get(self.y.iloc[index], np.nan)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.X)

class MultiCrop(data.Dataset):
    def __init__(self, dataset, cropmode, transform, seed):
        import re
        self.dataset = dataset
        self.num_crops, = map(int, re.findall(r'\d+', cropmode))
        self.cropmode = cropmode
        self.transform = transform
        self.seed = seed

    def __getitem__(self, index):
        orig_index, i = divmod(index, self.num_crops)
        img, target = self.dataset[orig_index]
        variant = self.cropmode[-1]
        if i > 0:
            random.seed((self.seed, index))
            if variant == 'a':
                img = transforms.CenterCrop(240)(img)
                img = transforms.RandomCrop(224)(img)
                img = transforms.RandomHorizontalFlip()(img)
                img = RandomVerticalFlip()(img)
                img = RandomTranspose()(img)
            else:
                raise ValueError("unknown variant")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataset) * self.num_crops

    @property
    def indices(self):
        return [index for index in self.dataset.indices for i in range(self.num_crops)]


# Generic transforms
# ==================

class RandomVerticalFlip(object):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomTranspose(object):
    """Transpose the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return img.transpose(Image.TRANSPOSE)
        return img
