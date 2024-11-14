import configparser

from model.Generator import Generator
from model.Discriminator import Discriminator
from utils.pix2pix_dataset import MapDataset
import torch
import torch.nn as nn
import utils.config
from utils.utils import *
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import os


def train(data_iter, G, D, G_trainer, D_trainer, G_scaler, D_scaler, bce_loss, l1_loss, lambda_D_HE, save_train=True):
    loop = tqdm(data_iter, leave=True)

    for idx, (x, y, H, E) in enumerate(loop):
        G_loss_save = 0

        x = x.to(config.DEVICE)
        y_HE = y.to(config.DEVICE)
        y_H = H.to(config.DEVICE)
        y_E = E.to(config.DEVICE)

        # train D
        with torch.cuda.amp.autocast():
            y_fake_H, y_fake_E, y_fake_HE = G(x)
            D_fake_H, D_fake_E, D_fake_HE = D(x, [y_fake_H.detach(), y_fake_E.detach(), y_fake_HE.detach()])
            D_real_H, D_real_E, D_real_HE = D(x, [y_H, y_E, y_HE])
            D_fake_H_loss = bce_loss(D_fake_H, torch.zeros_like(D_fake_H))
            D_real_H_loss = bce_loss(D_real_H, torch.ones_like(D_real_H))
            D_fake_E_loss = bce_loss(D_fake_E, torch.zeros_like(D_fake_E))
            D_real_E_loss = bce_loss(D_real_E, torch.ones_like(D_real_E))
            D_fake_HE_loss = bce_loss(D_fake_HE, torch.zeros_like(D_fake_HE))
            D_real_HE_loss = bce_loss(D_real_HE, torch.ones_like(D_real_HE))
            # D_loss = ((lambda_D_HE==0) * (D_real_H_loss + D_fake_H_loss + D_real_E_loss + D_fake_E_loss) + lambda_D_HE * (D_fake_HE_loss + D_real_HE_loss)) / 6
            D_loss = (D_real_H_loss + D_fake_H_loss + D_real_E_loss + D_fake_E_loss + lambda_D_HE * (D_fake_HE_loss + D_real_HE_loss)) / 6
            D_loss_real_save = D_real_H_loss + D_real_E_loss
            D_loss_fake_save = D_fake_H_loss + D_fake_E_loss

        D_trainer.zero_grad()
        D_scaler.scale(D_loss).backward()
        D_scaler.step(D_trainer)
        D_scaler.update()

        # train G
        with torch.cuda.amp.autocast():
            G_D_fake_H, G_D_fake_E, G_D_fake_HE = D(x, [y_fake_H, y_fake_E, y_fake_HE])
            G_D_fake_H_loss = bce_loss(G_D_fake_H, torch.ones_like(G_D_fake_H))
            G_D_fake_E_loss = bce_loss(G_D_fake_E, torch.ones_like(G_D_fake_E))
            G_D_fake_HE_loss = bce_loss(G_D_fake_HE, torch.ones_like(G_D_fake_HE))
            L1_H = l1_loss(y_fake_H, y_H)
            L1_E = l1_loss(y_fake_E, y_E)
            L1_HE = l1_loss(y_fake_HE, y_HE)

            G_H_loss = G_D_fake_H_loss + config.L1_H_LAMBDA * 2 * L1_H
            G_E_loss = G_D_fake_E_loss + config.L1_E_LAMBDA * 2 * L1_E
            G_HE_loss = G_D_fake_HE_loss + config.L1_HE_LAMBDA * 2 * L1_HE

            G_loss = G_H_loss + G_E_loss + G_HE_loss
            G_loss_save = (G_loss_save * idx + G_HE_loss) / (idx + 1)

        G_trainer.zero_grad()
        G_scaler.scale(G_loss).backward()
        G_scaler.step(G_trainer)
        G_scaler.update()

        # if idx % 20 == 0:
        #     loop.set_postfix(
        #         D_real=torch.sigmoid(D_real).mean().item(),
        #         D_fake=torch.sigmoid(D_fake).mean().item(),
        #         G_loss=G_loss.mean().item()
        #     )

    if save_train:
        return torch.sigmoid(D_loss_fake_save).mean().item(),  torch.sigmoid(D_loss_real_save).mean().item(), G_loss_save.mean().item()
    else:
        return 0, 0, 0


def main():
    # create models
    G = Generator().to(config.DEVICE)
    D = Discriminator().to(config.DEVICE)
    G_trainer = optim.Adam(G.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    # scheduler = lr_scheduler.MultiStepLR(G_trainer, milestones=[401, 410], gamma=0.5)
    D_trainer = optim.Adam(D.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    G_scaler = torch.cuda.amp.GradScaler()
    D_scaler = torch.cuda.amp.GradScaler()

    # create loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # if load pre-trained weights
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_dir + "model" + "_" + "0" + "_" + config.CHECKPOINT_GEN, G, G_trainer, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_dir + "model" + "_" + "0" + "_" + config.CHECKPOINT_DISC, D, D_trainer, config.LEARNING_RATE)

    # create Data_iter
    train_set = MapDataset(root_dir=config.TRAIN_DIR)
    val_set = MapDataset(root_dir=config.VAL_DIR)
    train_iter = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_iter = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    # train
    D_real_total, D_fake_total, G_loss_total = [], [], []
    training = True # ------------------IMPORTANT: train=True if want to train-----------------

    if training:
        for epoch in range(config.NUM_EPOCHS):
            print("------- epoch = " + str(epoch) + str("/") + str(config.NUM_EPOCHS-1) + " -------")
            # lambda_D_HE = 1
            if (epoch // 10) % 2 == 0:
            # if epoch % 2 == 0:
                print("---- training inner ----")
                lambda_D_HE = 0
                for param in G.CL.parameters():
                    param.requires_grad = False
                for param in D.D3.parameters():
                    param.requires_grad = False
                for param in D.D1.parameters():
                    param.requires_grad = True
                for param in D.D2.parameters():
                    param.requires_grad = True
            else:
                print("---- training outer ----")
                lambda_D_HE = 1
                for param in G.CL.parameters():
                    param.requires_grad = True
                for param in D.D3.parameters():
                    param.requires_grad = True
                for param in D.D1.parameters():
                    param.requires_grad = False
                for param in D.D2.parameters():
                    param.requires_grad = False

            # for param in G.DeBlur.layer1_decoder.parameters():
            #     print(param.grad)

            D_real, D_fake, G_loss = train(train_iter, G, D, G_trainer, D_trainer, G_scaler, D_scaler, bce_loss, l1_loss, lambda_D_HE=lambda_D_HE)
            # before_lr = G_trainer.param_groups[0]["lr"]
            # scheduler.step()
            # after_lr = G_trainer.param_groups[0]["lr"]
            # print("Epoch %d: SGD lr %.8f -> %.8f" % (epoch, before_lr, after_lr))

            D_real_total.append(D_real)
            D_fake_total.append(D_fake)
            G_loss_total.append(G_loss)

            # ---------------save single model:
            # if config.SAVE_MODEL and min(G_loss_total) == G_loss_total[-1]:
            #     save_checkpoint(G, G_trainer, filename=config.CHECKPOINT_GEN)
            #     save_checkpoint(D, D_trainer, filename=config.CHECKPOINT_DISC)

            # ---------------save n best models:
            if not os.path.exists(config.CHECKPOINT_dir):
                os.makedirs(config.CHECKPOINT_dir)

            save_model_num = 10
            if config.SAVE_MODEL:
                temp = G_loss_total[::]
                temp.sort()
                pos = temp.index(G_loss_total[-1])
                if pos < save_model_num:
                    save_checkpoint(G, G_trainer, filename=config.CHECKPOINT_dir + "model" + "_" + str(pos) + "_" + config.CHECKPOINT_GEN)
                    save_checkpoint(D, D_trainer, filename=config.CHECKPOINT_dir + "model" + "_" + str(pos) + "_" + config.CHECKPOINT_DISC)

            if not os.path.exists(config.SAVE_EXAMPLE_DIR):
                os.makedirs(config.SAVE_EXAMPLE_DIR)
            save_some_examples(G, val_iter, epoch, folder=config.SAVE_EXAMPLE_DIR, img_th=101)

        plot_train_process(D_real_total, D_fake_total, G_loss_total)
        df = pd.DataFrame(G_loss_total)
        df.to_excel(config.SAVE_LOSS_dir + "G_loss.xlsx")

    else:
        if not os.path.exists(config.TEST_dir):
                os.makedirs(config.TEST_dir)
        save_some_examples_for_test(G, val_iter, folder=config.TEST_dir)


if __name__ == "__main__":
    main()