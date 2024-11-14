from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from . import config


class MapDataset(Dataset):
    def __init__(self, root_dir, mednet=False):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.mednet = mednet
        # print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        if self.mednet:
            input_image = image[2:-2, 2:258, :]
            target_image = image[2:-2, 260:516, :]

            augmentations = config.both_transform(image=input_image, image0=target_image)
            input_image = augmentations["image"]
            target_image = augmentations["image0"]

            # aug2 = config.transform_only_input(image=input_image, image0=target_image)
            # input_image = aug2["image"]
            # target_image = aug2["image0"]
            input_image = config.transform_only_input(image=input_image)["image"]
            target_image = config.transform_only_mask(image=target_image)["image"]

            return input_image, target_image

        else:
            input_image = image[:, :256, :]
            target_image = image[:, 256:512, :]
            H_image = image[:, 512:768, :]
            E_image = image[:, 768:, :]

            augmentations = config.both_transform(image=input_image, image0=target_image, image1=H_image, image2=E_image)
            input_image = augmentations["image"]
            target_image = augmentations["image0"]
            H_image = augmentations["image1"]
            E_image = augmentations["image2"]

            # aug2 = config.transform_only_input(image=input_image, image0=target_image)
            # input_image = aug2["image"]
            # target_image = aug2["image0"]
            input_image = config.transform_only_input(image=input_image)["image"]
            target_image = config.transform_only_mask(image=target_image)["image"]
            H_image = config.transform_only_mask(image=H_image)["image"]
            E_image = config.transform_only_mask(image=E_image)["image"]

            return input_image, target_image, H_image, E_image


if __name__ == "__main__":
    dataset = MapDataset(config.TRAIN_DIR)
    loader = DataLoader(dataset, batch_size=10)
    for x, y, H, E in loader:
        print(x.shape)
        save_image(x[5:], "x.png")
        save_image(y[5:], "y.png")
        save_image(H[5:], "H.png")
        save_image(E[5:], "E.png")
        import sys

        sys.exit()
