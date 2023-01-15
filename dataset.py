import os

import numpy as np
import torch
from skimage import io, color
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    """
    Dataset from a directory of images
    """
    def __init__(self, directory: str, size: int, transform: bool):
        """
        Create a dataset from a directory of images.
        :param directory: location of images
        :param size: size to crop images to
        :param transform: true if images should be transformed for training
        """
        self.size = size

        paths = [x[0] for x in os.walk(directory)]
        self.filenames = []
        for path in paths:
            self.filenames += [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        # Random crop and horizontal flip for training, center crop for validation
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

            # Filter non-RGB images
            assert len(img.shape) == 3
            assert img.shape[-1] == 3
            assert img.shape[0] >= self.size
            assert img.shape[1] >= self.size

            img = self.transforms(img)

            # Convert to LAB
            img = torch.tensor(color.rgb2lab(img.permute(1, 2, 0)).astype(np.float32)).permute(2, 0, 1)

            return img
        except:
            return None

# When loading in the dataset, it is possible some images are non-RGB and thus we filter them out with the following colate function.
def collate_fn(batch):
    """
    Collate function for DataLoader that filters out None values
    :param batch: batch of images
    :return: batch of images
    """
    batch = [x for x in batch if x is not None]
    return torch.utils.data.dataloader.default_collate(batch)
