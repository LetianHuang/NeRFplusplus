"""
data_loader
===========
Provides
* Some datasets import functions for the NeRF project
---------------------------------------------------------
Author: LT H
Github: mofashaoye
"""
import json
import os

import cv2 as cv
import numpy as np


def load_blender(base_dir, resize_coef=2, background_w=True) -> dict:
    """
    Load the datasets whose type is blender
    =======================================
    Inputs:
        base_dir : str      The root directory of the datasets
        resize_coef : int   The transformation coefficient of the image size of the datasets
        background_w : bool Whether to remove the A channel from the RGBA image 
                            and set the background to white
    Outputs:
        datasets : dict     Datasets containing train set, validation set, and test set.
                            Each set contains three types of data, the image, 
                            the camera transformation matrix, and the focal length 
                            whose type are NumPy.NDArray
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        with open(os.path.join(base_dir, f"transforms_{split}.json"), "r") as f:
            dataset = json.load(f)
        images, poses = [], []
        for frame in dataset["frames"]:
            image = cv.imread(
                os.path.join(base_dir, frame["file_path"] + ".png"),
                cv.IMREAD_UNCHANGED
            )
            height, width, channel = image.shape
            if resize_coef >= 2:
                image = cv.resize(
                    image, (width // resize_coef, height // resize_coef), interpolation=cv.INTER_AREA
                )
            images.append(image)
            poses.append(np.array(frame["transform_matrix"]))
        images = np.array(images, dtype=np.float32) / 255.0
        if background_w:
            images = (images[..., :3] - 1.0) * images[..., -1:] + 1.0
        poses = np.array(poses, dtype=np.float32)
        # The focal length is calculated according to the field of view (FOV_x)
        # and the image width
        samples, height, width, channel = images.shape
        fov = float(dataset["camera_angle_x"])
        focal = 0.5 * width / np.tan(0.5 * fov) 
        
        datasets[split] = dict(images=images, poses=poses, focal=focal)
    return datasets


def load_realworld(base_dir, resize_coef=2, splits=["train", "validation", "test"]) -> dict:
    datasets = {}
    for split in splits:
        images, poses, intrinsics = [], [], []
        for root, dirs, files in os.walk(os.path.join(base_dir, split, "rgb")):
            files = sorted(files)
            for file in files:
                image = cv.imread(os.path.join(root, file), cv.IMREAD_UNCHANGED)
                height, width, channel = image.shape
                if resize_coef >= 2:
                    image = cv.resize(
                        image, (width // resize_coef, height // resize_coef), interpolation=cv.INTER_AREA
                    )    
                images.append(image)
        for root, dirs, files in os.walk(os.path.join(base_dir, split, "pose")):
            files = sorted(files)
            for file in files:
                pose = np.array(
                    list(map(lambda x : float(x), open(os.path.join(root, file)).read().split())), 
                    dtype=np.float32
                ).reshape((4, 4))
                poses.append(pose)
        for root, dirs, files in os.walk(os.path.join(base_dir, split, "intrinsics")):
            files = sorted(files)
            for file in files:
                intrinsic = np.array(
                    list(map(lambda x : float(x), open(os.path.join(root, file)).read().split())), 
                    dtype=np.float32
                ).reshape((4, 4))
                intrinsics.append(intrinsic)
        images = np.array(images, dtype=np.float32) / 255.0
        poses = np.array(poses, dtype=np.float32)
        intrinsics = np.array(intrinsics, dtype=np.float32)
        
        datasets[split] = dict(images=images, poses=poses, intrinsics=intrinsics)
    return datasets