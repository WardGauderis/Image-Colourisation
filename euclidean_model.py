import torch

from model import Model


class EuclideanModel(Model):
    """
    Colorization model with a Euclidean loss, derived from the default model
    """
    def __init__(self):
        # Only output 2 AB channels instead of a distribution over all Q values
        super(EuclideanModel, self).__init__("euclidean", 2, None, None, None)
        # Use MSE loss instead of cross entropy
        self.criterion = torch.nn.MSELoss()

    def encode(self, image: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Encode an image into L and (normalised) AB channels
        :param image: LAB image to encode
        :return:
        """
        image = image.to(self.device)
        l = image[:, :1, :, :]
        # Downsample the AB channels to 1/4 the size of the L channel as this is the output size of the model
        shape = (l.shape[-2] // 4, l.shape[-1] // 4)
        ab = torch.nn.functional.interpolate(image[:, 1:, :, :], size=shape, mode="bilinear")
        return l, ab / 110

    def decode(self, l: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        """
        Decode an image from L and (normalised) AB channels
        :param l: L channel
        :param ab: Normalised AB channels
        :return: LAB image
        """
        # Upsample the AB channels to the size of the L channel
        ab = torch.nn.functional.interpolate(ab * 110, size=l.shape[-2:], mode="bilinear")
        image = torch.cat((l, ab), dim=1)
        return image
