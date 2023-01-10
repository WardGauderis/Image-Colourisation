import torch

from model import Model


class EuclideanModel(Model):
    def __init__(self):
        super(EuclideanModel, self).__init__("euclidean", 2, None, None, None)
        self.criterion = torch.nn.MSELoss()

    def encode(self, image: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        image = image.to(self.device)
        l = image[:, :1, :, :]
        shape = (l.shape[-2] // 4, l.shape[-1] // 4)
        ab = torch.nn.functional.interpolate(image[:, 1:, :, :], size=shape, mode="bilinear")
        return l, ab / 110

    def decode(self, l: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        ab = torch.nn.functional.interpolate(ab * 110, size=l.shape[-2:], mode="bilinear")
        image = torch.cat((l, ab), dim=1)
        return image
