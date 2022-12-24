import os
import random

import numpy as np
import torch
import tqdm
from torch import nn

import data_loader
import nerf
import render

###################################################################
# mip-NeRF Training Hyperparameter
# (Use the same hyperparameters as the official implementation)
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
TRAIN_BATCH_SIZE = 1024
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


def get_random_screen_batch(
    height: int,
    width: int,
    train_batch_size: int,
    select_center: bool,
    device=torch.device("cpu"),
) -> torch.Tensor:
    """
    Get randome screen batch coordinates for training
    =================================================
    Inputs:
        height              : int                   scene's height
        width               : int                   scene's width
        train_batch_size    : int                   batch size of training
        select_center       : bool                  whether select center of the scene
        device              : torch.device          Output's device
    Output:
        coords              : torch.Tensor          batch coordinates of training
    """
    if select_center:
        dH = int(height // 2 * 0.5)
        dW = int(width // 2 * 0.5)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(height // 2 - dH, height // 2 +
                               dH - 1, 2 * dH, device=device),
                torch.linspace(width // 2 - dW, width // 2 +
                               dW - 1, 2 * dW, device=device)
            ),
            dim=-1
        )
    else:
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, height - 1, height, device=device),
                torch.linspace(0, width - 1, width, device=device)
            ),
            dim=-1
        )
    coords = torch.reshape(coords, (-1, 2))
    coords = coords[random.sample(
        list(range(coords.shape[0])), train_batch_size)].long()
    return coords


def train_nerfpp(
    datasets,
    fg_net: nn.Module,
    bg_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func,
    num_epochs,
    train_start_epoch=0
):
    if train_start_epoch == num_epochs:
        print("[Log] The model training is over. There is no need to continue training!")
        return
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
        is_train=True,
        device=DEVICE
    )
    loss_list = []
    if not os.path.exists("out"):
        os.mkdir("out")
    if not os.path.exists("./out/model"):
        os.mkdir("./out/model")
    if not os.path.exists("./out/imgs"):
        os.mkdir("./out/imgs")
    for epoch in tqdm.trange(train_start_epoch, num_epochs):
        renderer.train(True)
        # Obtain the epoch training data and do data migration (to GPU if have GPU)
        data_index = np.random.choice(list(range(len(images))))

        image = images[data_index]

        image = torch.tensor(image, device=DEVICE)
        pose = poses[data_index]
        intrinsic = intrinsics[data_index]
        
        coords = get_random_screen_batch(
                height, width, TRAIN_BATCH_SIZE, epoch <= 500, device=DEVICE
            )

        image = image[coords[..., 0], coords[..., 1]]
        # Volume renderer render to obtain predictive rgb (image)
        # Including ray generation, sample coordinates, positional encoding,
        # Hierarchical volume sampling, NeRF(x,d)=(rgb,density),
        # computation of volume rendering equation
        image_coarse, image_fine = renderer.render(pose, intrinsic, select_coords=coords)

        # Calculate the losses and use the optimizer for gradient descent backward
        optimizer.zero_grad()
        loss = loss_func(image, image_fine) + 0.1 * loss_func(image, image_coarse)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.detach().cpu().item())

        # Update learning rate
        decay_rate = 0.1
        decay_steps = LEARNING_RATE_DECLAY * 1000
        new_lrate = LEARNING_RATE * (decay_rate ** ((epoch + 1) / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if (epoch + 1) % EPOCH_PER_LOG == 0:
            # save NeRF model
            print(f"[Model] NeRF++ model {epoch + 1} saved successfully !")
            torch.save(renderer.fg_nerf.state_dict(), f"./out/model/fg_nerf_train_{epoch + 1}.pt")
            torch.save(renderer.bg_nerf.state_dict(), f"./out/model/bg_nerf_train_{epoch + 1}.pt")
            # save rendering image for test
            # See how well the model is trained and
            # see what the rendering image looks like
            print(f"[Render] NeRF++ render train_0 {epoch + 1} start !")
            img = renderer.render_image(poses[0], intrinsics[0])
            render.save_img(img, f"./out/imgs/mip_train_epoch_{epoch + 1}.png")
            print(f"[Render] NeRF++ render train_0 {epoch + 1} image saved successfully !")
            # The loss function is written to a log file
            # and then visualized with other module if need
            print(f"[Train] epoch={epoch + 1} Loss is {loss_list[-1]}")
            with open("./out/logs.txt", "a", encoding="utf-8") as f:
                f.writelines([str(x) + "\n" for x in loss_list])
                loss_list = []
                print(f"[Log] NeRF++ loss list saved successfully !")


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
    # get optimizer from torch.optim module
    optimizer = torch.optim.Adam(
        params=list(fg_net.parameters()) + list(bg_net.parameters()),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999)
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
                train_start_epoch = int(
                    result[len("fg_nerf_train_"):-len(".pt")])
                print(
                    f"[Train] fg-NeRF train start epoch is {train_start_epoch}")
        for root, dirs, files in os.walk("./out/model"):
            files = list(filter(lambda file: file.startswith("bg_nerf_train_") and file.endswith(".pt"), files))
            if files is not None and len(files) >= 1:
                result = max(files, key=lambda name: int(
                    name[len("bg_nerf_train_"):-len(".pt")]))
                path = os.path.join(root, result)
                print(f"[Model] bg-NeRF model Loader from {path}")
                bg_net.load_state_dict(torch.load(path))
                train_start_epoch = min(train_start_epoch, int(result[len("bg_nerf_train_"):-len(".pt")]))
                print(f"[Train] bg-NeRF train start epoch is {train_start_epoch}")
    # Start training
    train_nerfpp(
        datasets=datasets["train"],
        fg_net=fg_net,
        bg_net=bg_net,
        optimizer=optimizer,
        loss_func=loss_func,
        num_epochs=NUM_EPOCHS,
        train_start_epoch=train_start_epoch
    )
