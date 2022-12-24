import os

import numpy as np
import torch
import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from torch import nn

import data_loader
import nerf
import render

###################################################################
###################################################################
# OS parameters
DATA_BASE_DIR = "./data/tanks_and_temples/tanks_and_temples/tat_training_Truck/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
RESIZE_COEF = 1
BACKGROUND_W = True

# Model parameters
POS_ENCODE_DIM = 10
VIEW_ENCODE_DIM = 4
DENSE_FEATURES = 256
DENSE_DEPTH = 8

# Train parameters
TRAIN_BATCH_SIZE = 2048
LEARNING_RATE = 0.0005
LEARNING_RATE_DECLAY = 500
NUM_EPOCHS = 500000

# Render parameters
NUM_SAMPLES = 64
NUM_ISAMPLES = 128
RAY_CHUNK = 32768
SAMPLE5D_CHUNK = 65536

# Log parameters
EPOCH_PER_LOG = 2000
#############################################################################


def calcPSNR(img1, img2) -> float:
    return PSNR(img1, img2)


def calcSSIM(img1, img2) -> float:
    return SSIM(img1, img2, channel_axis=2)


def test_nerfpp(
    datasets,
    fg_net: nn.Module,
    bg_net: nn.Module,
    loss_func,
    split="test"
):
    images, poses, intrinsics = datasets["images"], datasets["poses"], datasets["intrinsics"]
    samples, height, width, channel = images.shape
    # Pass relevant scene parameters, camera parameters,
    # geometric model (NeRF) into Volume Renderer
    renderer = render.VolumeRendererPlusPlus(
        fg_nerf=fg_net,
        bg_nerf=bg_net,
        width=width,
        height=height,
        num_samples=NUM_SAMPLES,
        num_isamples=NUM_ISAMPLES,
        background_w=BACKGROUND_W,
        ray_chunk=RAY_CHUNK,
        sample5d_chunk=SAMPLE5D_CHUNK,
        is_train=False,
        device=DEVICE
    )
    if not os.path.exists("./out/other_imgs"):
        os.mkdir("./out/other_imgs")
    loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    print(f"[Log] number of {split}sets' images: {len(images)}")
    for epoch in tqdm.trange(0, len(images)):
        data_index = epoch
        
        image = images[data_index]
        pose = poses[data_index]
        intrinsic = intrinsics[data_index]

        # Volume renderer render to obtain predictive rgb (image)
        # Including ray generation, sample coordinates, positional encoding,
        # Hierarchical volume sampling, NeRF(x,d)=(rgb,density),
        # computation of volume rendering equation
        image_hat = renderer.render_image(pose, intrinsic, use_tqdm=False, render_batch_size=1024)

        render.save_img(
            image_hat, f"./out/other_imgs/{split}_hat_r_{data_index}.png"
        )

        # Calculate the loss and psnr
        loss = loss_func(
            torch.tensor(image),
            torch.tensor(image_hat)
        ).detach().item()
        
        loss_sum += loss
        psnr_sum += calcPSNR(image, image_hat)
        ssim_sum += calcSSIM(image, image_hat)

    loss_sum /= len(images)
    psnr_sum /= len(images)
    ssim_sum /= len(images)
    print(f"[Test] mip-NeRF avg Loss in {split}set: {loss_sum}")
    print(f"[Test] mip-NeRF avg PSNR in {split}set: {psnr_sum}")
    print(f"[Test] mip-NeRF avg SSIM in {split}set: {ssim_sum}")


if __name__ == "__main__":
    # get datasets from data_loader module
    datasets = data_loader.load_realworld(
        base_dir=DATA_BASE_DIR,
        resize_coef=RESIZE_COEF,
        splits=["train", "test"]
    )
    # get nerf from nerf module
    fg_net = nerf.NeRF(
        pos_dim=POS_ENCODE_DIM,
        view_dim=VIEW_ENCODE_DIM,
        dense_features=DENSE_FEATURES,
        dense_depth=DENSE_DEPTH,
        in_features=3
    )
    bg_net = nerf.NeRF(
        pos_dim=POS_ENCODE_DIM,
        view_dim=VIEW_ENCODE_DIM,
        dense_features=DENSE_FEATURES,
        dense_depth=DENSE_DEPTH,
        in_features=4
    )
    # get loss function from torch.nn module
    loss_func = nn.MSELoss(reduction="mean")
    # Read the model of the largest epoch ever trained from the logs folder
    train_start_epoch = 0
    if os.path.exists("./out") and os.path.exists("./out/model"):
        for root, dirs, files in os.walk("./out/model"):
            files = list(filter(lambda file: file.startswith("fg_nerf_train_") and file.endswith(".pt"), files))
            if files is not None and len(files) >= 1:
                result = max(files, key=lambda name: int(
                    name[len("fg_nerf_train_"):-len(".pt")]))
                path = os.path.join(root, result)
                print(f"[Model] fg-NeRF model Loader from {path}")
                fg_net.load_state_dict(torch.load(path))
        for root, dirs, files in os.walk("./out/model"):
            files = list(filter(lambda file: file.startswith("bg_nerf_train_") and file.endswith(".pt"), files))
            if files is not None and len(files) >= 1:
                result = max(files, key=lambda name: int(
                    name[len("bg_nerf_train_"):-len(".pt")]))
                path = os.path.join(root, result)
                print(f"[Model] bg-NeRF model Loader from {path}")
                bg_net.load_state_dict(torch.load(path))
    # Start training
    test_nerfpp(
        datasets=datasets["test"],
        fg_net=fg_net,
        bg_net=bg_net,
        loss_func=loss_func,
        split="test"
    )

