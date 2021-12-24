# DataSet
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode

def _norm_advprop(img):
    return img * 2.0 - 1.0

def build_transform(dest_image_size):
    normalize = transforms.Lambda(_norm_advprop)
    if not isinstance(dest_image_size, tuple):
        dest_image_size = (dest_image_size, dest_image_size)
    else:
        dest_image_size = dest_image_size

    transform = transforms.Compose([
        transforms.Resize(dest_image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize
    ])

    return transform


def build_data_set(dest_image_size, data):
    transform = build_transform(dest_image_size)
    dataset = datasets.ImageFolder(data, transform=transform, target_transform=None)
    return dataset
