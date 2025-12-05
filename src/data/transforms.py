import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.03, p=1.0),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.ImageCompression(quality_range=(9, 100), p=1.0),
        ], p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(mean=[0.5631, 0.4959, 0.7355], std=[0.2419, 0.2835, 0.1761]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5631, 0.4959, 0.7355], std=[0.2419, 0.2835, 0.1761]),
        ToTensorV2(),
    ])