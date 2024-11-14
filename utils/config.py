import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
LEARNING_RATE = 2e-4
# LEARNING_RATE_MEDNET = 8e-4
# LEARNING_RATE_MEDNET_G = 2e-5
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_HE_LAMBDA = 100
L1_H_LAMBDA = 100
L1_E_LAMBDA = 50
LAMBDA_GP = 10
NUM_EPOCHS = 400
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_dir = "model/checkpoints/"
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
SAVE_EXAMPLE_DIR = "results/examples"
SAVE_LOSS_dir = "results/"
TEST_dir = "results/test/"

both_transform = A.Compose(
    [A.Resize(width=256, height=256), A.HorizontalFlip(p=0.5)],
    additional_targets={"image0": "image",
                        "image1": "image",
                        "image2": "image"},
)

# transform_only_input = A.Compose(
#     [
#         A.HorizontalFlip(p=0.5),
#         A.ColorJitter(p=0.2),
#         A.Normalize(mean=[.5, .5, .5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
#         ToTensorV2(),
#     ],
#     additional_targets={"image0": "image"},
# )

transform_only_input = A.Compose(
    [
        # A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[.5, .5, .5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
