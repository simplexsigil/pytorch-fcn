import os
import os.path as osp
import random as rnd
import sys

import PIL.Image
import numpy as np
import png
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange


def build_path(root_path, f_dir, img_id, ext): return osp.join(root_path, f_dir, img_id + ext)


def main():
    rnd.seed(42)

    root_path = r"/home/david/Daten/datasets/cmu-airlab/assignment-task-5/data"
    paths = {
        "imgs": "left",
        "disps": "disp",
        "lbls": "labeled",
        "aug_imgs": "left/augmented",
        "aug_disps": "disp/augmented",
        "aug_lbls": "labeled/augmented"
    }
    ds_ext = {
        "imgs": "_left.jpg",
        "disps": "_disp.png",
        "lbls": "_left.png",
        "aug_imgs": "_aug_left.jpg",
        "aug_disps": "_aug_disp.png",
        "aug_lbls": "_aug_left.png"}

    lbl_img_ids = extract_ids(osp.join(root_path, paths["lbls"]), augmented=False)

    make_path(osp.join(root_path, paths["aug_imgs"]))
    make_path(osp.join(root_path, paths["aug_disps"]))
    make_path(osp.join(root_path, paths["aug_lbls"]))

    # for each id, load originals and apply transformations
    for img_id in tqdm(lbl_img_ids, ncols=80, leave=False, file=sys.stdout):
        file_paths = {
            "img": osp.join(root_path, paths["imgs"], img_id + ds_ext["imgs"]),
            "disp": osp.join(root_path, paths["disps"], img_id + ds_ext["disps"]),
            "lbl": osp.join(root_path, paths["lbls"], img_id + ds_ext["lbls"])
        }

        # load image
        img_file = file_paths["img"]
        org_img = PIL.Image.open(img_file)

        # load disparity
        with open(file_paths["disp"], mode="rb") as disp_file:  # Use file to refer to the file object
            disp_reader = png.Reader(disp_file)
            disp_data = disp_reader.asDirect()
            disp_pixels = disp_data[2]
            org_disp = []
            for row in disp_pixels:
                row = np.asarray(row, dtype=np.uint16)
                org_disp.append(row)
            org_disp = np.stack(org_disp, 1)
            org_disp = org_disp.transpose()

        org_disp = PIL.Image.fromarray(org_disp)

        # load label
        lbl_file = file_paths["lbl"]
        org_lbl = PIL.Image.open(lbl_file)

        org_lbl_np = np.array(org_lbl, dtype=np.int32)
        min_lbl = np.amin(org_lbl_np)
        max_lbl = np.amax(org_lbl_np)


        for i in trange(25, ncols=80, leave=False, file=sys.stdout):
            new_id = img_id + '{:02d}'.format(i)
            img, disp, lbl = apply_random_transform(org_img, org_disp, org_lbl)

            lbl_np = np.array(lbl, dtype=np.int32)
            assert min_lbl <= np.amin(lbl_np)
            assert max_lbl >= np.amax(lbl_np)

            img.save(osp.join(root_path, paths["aug_imgs"], new_id + ds_ext["aug_imgs"]))
            lbl.save(osp.join(root_path, paths["aug_lbls"], new_id + ds_ext["aug_lbls"]))

            # Disparity augmentation not working yet, because TF transform changes mode from L;16 to RGB
            disp = np.asarray(disp, dtype=np.uint16)
            disp = png.from_array(disp, mode="L;16")
            disp = disp.save(osp.join(root_path, paths["aug_disps"], new_id + ds_ext["aug_disps"]))

        # img_dir = osp.join(root_path, path)

        # https://pytorch.org/docs/stable/torchvision/transforms.html
        # trf.adjust_brightness(None, brightness_factor=1)
        # .adjust_contrast(img, contrast_factor)
        # .adjust_gamma(img, gamma, gain=1)
        # .affine(img, angle, translate, scale, shear, resample=0,fillcolor=None)
        # .center_crop(img, output_size)
        # .crop(img, top, left, height, width)
        # .hflip(img)
        # .normalize(tensor, mean, std, inplace=False)
        # ..pad(img, padding, fill=0, padding_mode='constant')
        # .resize(img, size, interpolation=2)
        # .rotate


def make_path(dirName):
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")


def apply_random_transform(img, disp, lbl):
    if rnd.random() > 0.5:
        img = TF.hflip(img)
        disp = TF.hflip(disp)
        lbl = TF.hflip(lbl)

    angle = rnd.random() * 180 - 90
    translate1 = rnd.random() * 50 - 25
    translate2 = rnd.random() * 50 - 25
    translate = (translate1, translate2)
    scale = rnd.random() + 1
    shear = rnd.random() * 20 - 10

    img = TF.affine(img, angle, translate, scale, shear, resample=PIL.Image.BICUBIC)
    disp = TF.affine(disp, angle, translate, scale, shear, resample=PIL.Image.NEAREST)
    lbl = TF.affine(lbl, angle, translate, scale, shear, resample=PIL.Image.NEAREST)

    img = TF.adjust_brightness(img, 0.5 + rnd.random())

    img = TF.center_crop(img, output_size=(512, 640))
    disp = TF.center_crop(disp, output_size=(512, 640))
    lbl = TF.center_crop(lbl, output_size=(512, 640))

    return img, disp, lbl


def extract_ids(img_dir, augmented=False):
    if not augmented:
        lbl_ids = [f[0:4] for f in os.listdir(img_dir) if osp.isfile(osp.join(img_dir, f))]
    else:
        lbl_ids = [f[0:7] for f in os.listdir(img_dir) if osp.isfile(osp.join(img_dir, f))]
    return lbl_ids


if __name__ == '__main__':
    main()
