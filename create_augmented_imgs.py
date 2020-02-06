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

    root_path = osp.expanduser('~/Daten/datasets/cmu-airlab/assignment-task-5/data')
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

        for i in trange(15, ncols=80, leave=False, file=sys.stdout):
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


def make_path(dirName):
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")


def apply_random_transform(img: PIL.Image, disp, lbl):
    # All available transforms:
    # https://pytorch.org/docs/stable/torchvision/transforms.html

    if rnd.random() > 0.2:
        img = TF.hflip(img)
        disp = TF.hflip(disp)
        lbl = TF.hflip(lbl)

    angle = rnd.random() * 30 - 15
    translate = (0, 0)
    scale = rnd.random() * 0.5 + 1
    shear = rnd.random() * 10 - 5

    # Ensures we scale at least enough to avoid black background from rotation.
    scale = max(scale, rotation_scaling(img.size, angle))

    img = TF.affine(img, angle, translate, scale, shear, resample=PIL.Image.BICUBIC)
    disp = TF.affine(disp, angle, translate, scale, shear, resample=PIL.Image.NEAREST)
    # Resampling must be nearest, anything else does not make sense for labels.
    lbl = TF.affine(lbl, angle, translate, scale, shear, resample=PIL.Image.NEAREST)

    img = TF.adjust_brightness(img, 0.75 + rnd.random() * 0.5)

    img = TF.center_crop(img, output_size=(512, 640))
    disp = TF.center_crop(disp, output_size=(512, 640))
    lbl = TF.center_crop(lbl, output_size=(512, 640))

    img_cropped = TF.five_crop(img, size=(512, 640))
    disp_cropped = TF.five_crop(disp, size=(512, 640))
    lbl_cropped = TF.five_crop(lbl, size=(512, 640))

    imgs = [img]
    imgs.extend(img_cropped)
    disps = [disp]
    disps.extend(disp_cropped)
    lbls = [lbl]
    lbls.extend(lbl_cropped)

    idx = rnd.choice(range(len(imgs)))

    return imgs[idx], disps[idx], lbls[idx]


def rotation_scaling(size, angle):
    # When we rotate the corners, along some axis (x or y) they are forther away from origin than before.
    # We can take the x and y values and divide them by the original ones to determine a necessary scaling.
    # For any corner: Distance to rot. center is distance of point (width/2, height/2) to origin.
    # We can assume x and y axis to be symmetric, so we can work with positive values only.

    point = size[0] / 2., size[1] / 2.
    rot_point = rotate_point(point, angle)
    rot_point = abs(rot_point[0]), abs(rot_point[1])  # We assume x and y to be symmetric and only check on one point.

    # Determine scaling, so original point is at least as far from origin along each axis as rotated one.
    rot_s = np.divide(rot_point, point)
    rot_scaling = rot_s[0] if rot_s[0] > 1 else 1 / rot_s[0], rot_s[1] if rot_s[1] > 1 else 1 / rot_s[1]
    rot_scaling = max(rot_scaling)
    return rot_scaling


def rotate_point(point, angle_deg):
    import math
    phi = angle_deg / 360 * 2 * math.pi
    x, y = point

    return math.cos(phi) * x - math.sin(phi) * y, math.sin(phi) * x + math.cos(phi) * y


def extract_ids(img_dir, augmented=False):
    if not augmented:
        lbl_ids = [f[0:4] for f in os.listdir(img_dir) if osp.isfile(osp.join(img_dir, f))]
    else:
        lbl_ids = [f[0:7] for f in os.listdir(img_dir) if osp.isfile(osp.join(img_dir, f))]
    return lbl_ids


if __name__ == '__main__':
    main()
