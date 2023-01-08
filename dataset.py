import torch
import os
import random
import numpy as np
from skimage import io, color

from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, size: int, transform: bool):
        self.size = size
        self.count = 0

        paths = [x[0] for x in os.walk("DATA")]

        self.filenames = []
        for path in paths:
            self.filenames += [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        if transform:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(size),
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        try:
            img = io.imread(self.filenames[idx])

            assert len(img.shape) == 3
            assert img.shape[-1] == 3
            assert img.shape[0] >= self.size
            assert img.shape[1] >= self.size

            img = self.transforms(img)

            img = torch.tensor(color.rgb2lab(img.permute(1, 2, 0)).astype(np.float32)).permute(2, 0, 1)

            return img
        except:
            self.count += 1
            return None


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return torch.utils.data.dataloader.default_collate(batch)
