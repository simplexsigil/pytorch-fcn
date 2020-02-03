import os
import os.path as osp
import random

import PIL.Image
import numpy as np
import png
import torch
from torch.utils import data


class AirLabClassSegBase(data.Dataset):
    class_names = np.array([
        'background',
        'building',
        'dirt',
        'foliage',
        'grass',
        'human',
        'pole',
        'rails',
        'road',
        'sign',
        'sky'
    ])

    # This is probably not correct for this dataset, but lets see how it goes.
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, val=False, transform=True, k_fold=4, k_fold_val=0, shuffle=True, shuffle_seed=42,
                 max_len=None, use_augmented=True):
        self.root = root
        self._transform = transform
        self.val = val
        self.max_len = max_len

        # Creating lists of images, disparities and labels
        self.dataset_dir = osp.join(self.root, 'cmu-airlab/assignment-task-5/data')

        self.img_dir = osp.join(self.dataset_dir, "left")
        self.disp_dir = osp.join(self.dataset_dir, "disp")
        self.lbl_dir = osp.join(self.dataset_dir, "labeled")
        self.ds_ext = {"img": "_left.jpg", "disp": "_disp.png", "lbl": "_left.png"}

        # Preparing the datasets. Only maintaining ids.
        self.img_ids = [f[0:4] for f in os.listdir(self.img_dir) if osp.isfile(osp.join(self.img_dir, f))]
        self.disp_ids = [f[0:4] for f in os.listdir(self.disp_dir) if osp.isfile(osp.join(self.disp_dir, f))]
        self.lbl_ids = [f[0:4] for f in os.listdir(self.lbl_dir) if osp.isfile(osp.join(self.lbl_dir, f))]

        # Selecting cross validation data.
        val_start = int(len(self.lbl_ids) / k_fold * k_fold_val)
        val_end = int(len(self.lbl_ids) / k_fold * (k_fold_val + 1))

        # Preparing id prefixes of the train val set. Will be use to select augmented data as well.
        self.lbl_id_prefs_train = self.lbl_ids[:val_start]
        self.lbl_id_prefs_train.extend(self.lbl_ids[val_end:])
        self.lbl_id_prefs_val = self.lbl_ids[val_start:val_end]

        if use_augmented:
            self.img_dir = osp.join(self.dataset_dir, "left/augmented")
            self.disp_dir = osp.join(self.dataset_dir, "disp/augmented")
            self.lbl_dir = osp.join(self.dataset_dir, "labeled/augmented")
            self.ds_ext = {"img": "_aug_left.jpg", "disp": "_aug_disp.png", "lbl": "_aug_left.png"}

            # Preparing the datasets. Only maintaining ids.
            self.img_ids = [f[0:6] for f in os.listdir(self.img_dir) if osp.isfile(osp.join(self.img_dir, f))]
            self.disp_ids = [f[0:6] for f in os.listdir(self.disp_dir) if osp.isfile(osp.join(self.disp_dir, f))]
            self.lbl_ids = [f[0:6] for f in os.listdir(self.lbl_dir) if osp.isfile(osp.join(self.lbl_dir, f))]

            # Only use augemented images which correlate to original train val folds. Id string starts with the same id.
            self.lbl_id_prefs_train = [f for f in self.lbl_ids if f[0:4] in self.lbl_id_prefs_train]
            self.lbl_id_prefs_val = [f for f in self.lbl_ids if f[0:4] in self.lbl_id_prefs_val]

        if shuffle:
            random.seed(shuffle_seed)
            random.shuffle(self.lbl_ids)

        print("Dataset contains {} labelled images (augmentation={})".format(len(self.lbl_ids), str(use_augmented)))

    def build_path(self, f_dir, img_id, ext): return osp.join(f_dir, img_id + ext)

    def __len__(self):
        if self.val:
            return len(self.lbl_id_prefs_val) if self.max_len is None else self.max_len
        else:
            return len(self.lbl_id_prefs_train) if self.max_len is None else self.max_len

    def __getitem__(self, index):
        if self.val:
            file_id = self.lbl_id_prefs_val[index]
        else:
            file_id = self.lbl_id_prefs_train[index]

        paths = (self.build_path(self.img_dir, file_id, self.ds_ext["img"]),
                 self.build_path(self.disp_dir, file_id, self.ds_ext["disp"]),
                 self.build_path(self.lbl_dir, file_id, self.ds_ext["lbl"]))
        # load image
        img_file = paths[0]
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)

        # load disparity
        with open(paths[1], mode="rb") as disp_file:  # Use file to refer to the file object
            disp_reader = png.Reader(disp_file)
            disp_data = disp_reader.asDirect()
            disp_pixels = disp_data[2]
            disp = []
            for row in disp_pixels:
                row = np.asarray(row, dtype=np.uint16)
                disp.append(row)
            disp = np.stack(disp, 1)
            disp = disp.transpose()

        # load label
        lbl_file = paths[2]
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1

        if np.any(lbl == -1):
            print("Test: Label was -1.")

        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)

        # disp = disp.astype(np.float64)

        img = torch.from_numpy(img).float()
        # disp = torch.from_numpy(disp).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def untransform(self, img, lbl):
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]

        # disp = disp.numpy()
        # disp = disp.astype(np.uint16)

        return img, lbl
