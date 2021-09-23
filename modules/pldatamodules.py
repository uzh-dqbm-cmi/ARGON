import torch
import numpy as np
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import (
    IuxrayMultiImageDataset,
    )

image_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PadSquare:
    def __call__(self, img):
        w, h = img.size
        diff = abs(w - h)
        p1 = diff // 2
        p2 = p1
        if diff % 2 == 1:
            p2 += 1
        if w > h:
            return transforms.functional.pad(img, (0, p1, 0, p2))
        else:
            return transforms.functional.pad(img, (p1, 0, p2, 0))

    def __repr__(self):
        return self.__class__.__name__

def transform_image(split, args=None):
    if split == 'train' or split == args.folds.split(","):
        return(
            transforms.Compose([
            transforms.Resize(256),  # 256
            transforms.RandomCrop(256),  # 256
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        )

    else:
        # self.batch_size = self.batch_size * self.args.n_gpu
        return (
        transforms.Compose([
            transforms.Resize(256),  # 256
            transforms.CenterCrop(256),  # 256
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        )

def transform_image_ifcc(split, args=None):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rotate = transforms.RandomApply([transforms.RandomRotation(10.0, expand=True)], p=0.5)
    color = transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5)
    if split == 'train' or split == args.folds.split(","):
        augs = [rotate, color]
        augs += [PadSquare(),transforms.Resize(224)]
    else:
        augs =[]
    return (
        transforms.Compose(
            [PadSquare(),
             transforms.Resize(224)] +
            augs + [transforms.ToTensor(),
                    norm]
        )
    )

class DataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        #self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        # self.transform=image_pipeline # if the extractor is freez
        self.N = None

    def setup(self, stage=None):
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.
        self.train= IuxrayMultiImageDataset(self.args, self.tokenizer, 'train', transform=transform_image_ifcc('train', self.args),
                                                   limit_length=self.N)
        self.validation = IuxrayMultiImageDataset(self.args, self.tokenizer, 'val', transform=transform_image_ifcc('val', self.args),
                                             limit_length=self.N)
        self.test = IuxrayMultiImageDataset(self.args, self.tokenizer, 'test', transform=transform_image_ifcc('test', self.args),
                                             limit_length=self.N)
    def train_dataloader(self):
        return DataLoader(self.train, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.validation,collate_fn=self.collate_fn, batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.test,collate_fn=self.collate_fn, batch_size=self.batch_size)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)

        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return torch.LongTensor(images_id), images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)
