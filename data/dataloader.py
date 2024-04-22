import torch
import torchvision
from lightly.data import LightlyDataset
from lightly.transforms import MoCoV2Transform, utils
from torch.utils.data import Subset
import numpy as np
import pytorch_lightning as pl

def get_dataloaders(path_to_train, path_to_test, batch_size, num_workers, use_toy_dataset=True, toy_dataset_size=20000):

    # Augmentations typically used to train on cifar-10
    train_classifier_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=utils.IMAGENET_NORMALIZE["mean"],
                std=utils.IMAGENET_NORMALIZE["std"],
            ),
        ]
    )

    # No additional augmentations for the test set
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=utils.IMAGENET_NORMALIZE["mean"],
                std=utils.IMAGENET_NORMALIZE["std"],
            ),
        ]
    )

    # We use the moco augmentations for training moco
    transform = MoCoV2Transform(
        input_size=32,
        gaussian_blur=0.0,
    )
    dataset_train_moco = LightlyDataset(input_dir=path_to_train, transform= transform)

    # Since we also train a linear classifier on the pre-trained moco model we
    # reuse the test augmentations here (MoCo augmentations are very strong and
    # usually reduce accuracy of models which are not used for contrastive learning.
    # Our linear layer will be trained using cross entropy loss and labels provided
    # by the dataset. Therefore we chose light augmentations.)
    dataset_train_classifier = LightlyDataset(
        input_dir=path_to_train, transform=train_classifier_transforms
    )

    dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)

    print(len(dataset_train_moco), len(dataset_train_classifier), len(dataset_test))

    if use_toy_dataset:
        indices_train = np.random.choice(range(len(dataset_train_moco)), toy_dataset_size, replace=False)
        indices_test = np.random.choice(range(len(dataset_test)), int(toy_dataset_size / 3), replace=False)
        
        dataset_train_moco = Subset(dataset_train_moco, indices_train)
        dataset_train_classifier = Subset(dataset_train_classifier, indices_train)
        dataset_test = Subset(dataset_test, indices_test)

        print("\tTOY Dataset sizes:")
        print("Train MoCo:", len(dataset_train_moco))
        print("Train Classifier:", len(dataset_train_classifier))
        print("Test:", len(dataset_test))

    dataloader_train_moco = torch.utils.data.DataLoader(
        dataset_train_moco,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    dataloader_train_classifier = torch.utils.data.DataLoader(
        dataset_train_classifier,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    
    return dataloader_train_moco, dataloader_train_classifier, dataloader_test