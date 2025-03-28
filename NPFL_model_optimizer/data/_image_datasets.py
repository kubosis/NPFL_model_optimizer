#!/usr/bin/env python3
# 3b65dab7-9b72-43fc-90c5-9bbbfb304ea9
# 3037cdec-8857-4d1e-8707-4b3916e39158

import random

import numpy as np
import torch

import npfl138

def rand_bbox(size, lam):
    """ Generate a random bounding box with area proportional to lambda """
    W = size[2]
    H = size[1]
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))

    # Select a random center position
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Calculate bounding box coordinates
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2

class BaseDataset(npfl138.TransformedDataset):
    def __init__(self, dataset, base_transform):
        super().__init__(dataset)
        self._transform = base_transform

    def transform(self, example):
        image = example["image"]
        image = image.to(torch.float32)
        image = self._transform(image)
        label = example["label"]
        return image, label  # return an (input, target) pair

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.augmentations = 5

    def __len__(self):
        return len(self.dataset) * self.augmentations

    def __getitem__(self, idx):
        base_idx = idx // self.augmentations  # Get the base image
        aug_type = idx % self.augmentations  # Determine augmentation type

        image, label = self.dataset[base_idx]

        # Only apply transform if it exists and image hasn't been transformed yet
        if self.transform:
            image = self.transform(image)

        if aug_type == 0:
            # Original image
            pass

        elif aug_type == 1:
            # Horizontal flip (standard for CIFAR-10)
            image = torch.flip(image, dims=[2])

        elif aug_type == 2:
            # Random noise addition
            noise = torch.randn_like(image) * 0.025
            image = image + noise

        elif aug_type == 3:
            # Color jitter - adjust brightness
            brightness_factor = torch.empty(1).uniform_(0.8, 1.2).item()
            image = image * brightness_factor
            image = torch.clamp(image, 0, 1) if image.max() > 1 else image

        elif aug_type == 4:
            # Random rotation (small angle)
            if image.shape[0] == 3:
                angle = torch.empty(1).uniform_(-15, 15).item()
                theta = torch.tensor([
                    [torch.cos(torch.tensor(angle * 3.14159 / 180)), -torch.sin(torch.tensor(angle * 3.14159 / 180)),
                     0],
                    [torch.sin(torch.tensor(angle * 3.14159 / 180)), torch.cos(torch.tensor(angle * 3.14159 / 180)), 0]
                ], dtype=torch.float)

                # Apply affine grid and grid sample
                grid = torch.nn.functional.affine_grid(
                    theta.unsqueeze(0),
                    torch.Size((1, image.shape[0], image.shape[1], image.shape[2])),
                    align_corners=False
                )
                image = torch.nn.functional.grid_sample(
                    image.unsqueeze(0),
                    grid,
                    align_corners=False
                ).squeeze(0)
        return image, label


class CutMixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes=10, prob=0.5, smoothing=0.1, transform=None):
        self.dataset = dataset
        self.num_classes = num_classes
        self.prob = prob # probability of cutmix
        self.smoothing = smoothing  # Label smoothing parameter
        self.transform = transform

    def smooth_one_hot(self, label, num_classes):
        """ Convert label to one-hot with smoothing """
        one_hot = torch.full((num_classes,), self.smoothing / (num_classes - 1))
        one_hot[label.cpu().item()] = 1 - self.smoothing
        return one_hot

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]

        if self.transform:
            img1 = self.transform(img1)

        #label1 = self.smooth_one_hot(label1, self.num_classes)
        label1 = torch.nn.functional.one_hot(label1.long(), self.num_classes).float()

        if np.random.rand() < self.prob:
            idx2 = random.randint(0, len(self.dataset) - 1)
            img2, label2 = self.dataset[idx2]

            if self.transform:
                img2 = self.transform(img2)

            #label2 = self.smooth_one_hot(label2, self.num_classes)
            label2 = torch.nn.functional.one_hot(label2.long(), self.num_classes).float()
            # Sample lambda from Beta distribution
            lam = np.random.uniform(0.1, 0.45)
            x1, y1, x2, y2 = rand_bbox(img1.shape, lam)
            img1[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]

            # Adjust lambda based on actual pixel region
            lam = 1 - ((x2 - x1) * (y2 - y1) / (img1.shape[1] * img1.shape[2]))
            # Mixed label
            label1 = lam * label1 + (1 - lam) * label2

        return img1, label1

# TODO add RandAugment
