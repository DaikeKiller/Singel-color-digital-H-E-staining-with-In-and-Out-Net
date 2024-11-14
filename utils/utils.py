import os.path

import torch
from . import config
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance
import numpy as np


def save_some_examples(gen, val_loader, epoch, folder, img_th, mednet=False):
    if not os.path.exists(folder):
        os.mkdir(folder)

    loop = tqdm(val_loader, leave=True)

    count = 0
    for (a, b, c, d) in loop:
        if count > img_th:
            break
        x, y, H, E = a, b, c, d
        count += 1
    x, y, H, E = x.to(config.DEVICE), y.to(config.DEVICE), H.to(config.DEVICE), E.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_H_fake, y_E_fake, y_HE_fake = gen(x)[0] if mednet else gen(x)
        y_H_fake = y_H_fake * 0.5 + 0.5  # remove normalization#
        y_E_fake = y_E_fake * 0.5 + 0.5
        y_HE_fake = y_HE_fake * 0.5 + 0.5
        out = torch.cat((y_H_fake, y_E_fake, y_HE_fake), dim=3)
        # print(y_HE_out)
        save_image(out, folder + f"/y_gen_{epoch}.png")
        if epoch == 0:
            save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
            save_image(H * 0.5 + 0.5, folder + f"/label_H_{epoch}.png")
            save_image(E * 0.5 + 0.5, folder + f"/label_E_{epoch}.png")
    gen.train()


def save_some_examples_mednet(gen, val_loader, epoch, folder, img_th):
    if not os.path.exists(folder):
        os.mkdir(folder)

    loop = tqdm(val_loader, leave=True)

    count = 0
    for (a, b) in loop:
        if count > img_th:
            break
        x, y = a, b
        count += 1
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake_1, y_fake_2 = gen(x)
        y_fake_1 = y_fake_1 * 0.5 + 0.5  # remove normalization#
        y_fake_2 = y_fake_2 * 0.5 + 0.5
        out = y_fake_1
        # print(y_HE_out)
        save_image(out, folder + f"/y_gen_{epoch}.png")
        if epoch == 0:
            save_image(x, folder + f"/input_{epoch}.png")
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_some_examples_for_test(gen, val_loader, folder, mednet=False):
    if not os.path.exists(folder):
        os.makedirs(folder)

    loop = tqdm(val_loader, leave=True)

    count = 0
    for (a, b, c, d) in loop:
        x, y, H, E = a, b, c, d
        x, y, H, E = x.to(config.DEVICE), y.to(config.DEVICE), H.to(config.DEVICE), E.to(config.DEVICE)
        gen.eval()
        with torch.no_grad():
            H_fake, E_fake, y_fake = gen(x)[0] if mednet else gen(x)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            H_fake = H_fake * 0.5 + 0.5
            E_fake = E_fake * 0.5 + 0.5
            # alpha = 0.9  # Contrast control (1.0-3.0)
            # beta = 1  # Brightness control (0-100)
            # y_fake = cv2.addWeighted(np.array(y_fake.cpu()), alpha, np.zeros(y_fake.shape, np.array(y_fake.cpu()).dtype), 0, beta)
            # img = torch.cat([x * 0.5 + 0.5, torch.tensor(y_fake).to(config.DEVICE), y * 0.5 + 0.5], 0)
            # y_fake = F.adjust_saturation(y_fake, 1.3)
            # y_fake = F.adjust_hue(y_fake, -0.01)
            # y_fake = F.adjust_contrast(y_fake, 1.02)
            img = torch.cat([x * 0.5 + 0.5, y_fake, y * 0.5 + 0.5], 3)
            # print(img.shape)
            save_image(img, folder + f"/result{count}.png")
        # gen.train()
        count += 1


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def plot_train_process(D_real_total, D_fake_total, G_loss_total):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(list(range(1, len(D_real_total) + 1)), D_real_total)
    plt.plot(list(range(1, len(D_fake_total) + 1)), D_fake_total)
    plt.legend(["D_real", "D_fake"])
    plt.subplot(1, 2, 2)
    plt.plot(list(range(1, len(G_loss_total) + 1)), G_loss_total)
    plt.legend(["G_loss"])
    plt.show()


def plot_train_process_mednet(loss_total):
    plt.figure()
    plt.plot(list(range(1, len(loss_total) + 1)), loss_total)
    plt.legend(["G_loss"])
    plt.show()


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # print(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_checkpoint_for_test(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # print(checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=False)


def color_decom(imgs): # image size [batch_size, 3, img_size, img_size]
    batch_size = imgs.shape[0]
    img_size = imgs.shape[2]
    Hs = torch.zeros((batch_size, 3, img_size, img_size))
    Es = torch.zeros((batch_size, 3, img_size, img_size))
    Hs = Hs.to(config.DEVICE)
    Es = Es.to(config.DEVICE)
    v = torch.Tensor([[-0.5, -0, 1], [-0.45, -0.46, 1], [-0, -0.12, 1]])
    v_inv = torch.linalg.inv(v)
    v_inv = v_inv.to(config.DEVICE)
    for i in range(batch_size):
        img = imgs[i, :, :, :]
        img_flat = img.reshape((3, -1))
        s = torch.matmul(v_inv, img_flat)
        H = s[0, :]
        E = s[1, :]
        H = H.reshape((img_size, img_size))
        E = E.reshape((img_size, img_size))
        Hs[i, 0, :, :] = H
        Es[i, 0, :, :] = E
        Hs[i, 1, :, :] = H
        Es[i, 1, :, :] = E
        Hs[i, 2, :, :] = H
        Es[i, 2, :, :] = E
    return Hs, Es
