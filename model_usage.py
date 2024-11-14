# This file is to load the trained model and make the output on the data of interest. (only Generator)
import torch

import utils.config
from model.Generator import Generator as G
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
from skimage.measure import block_reduce


def load_model(model_name, model_type):
    # create models
    G = model_type().to(config.DEVICE)
    G_trainer = optim.Adam(G.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    load_checkpoint_for_test(model_name, G, G_trainer, config.LEARNING_RATE)

    return G


def use_model(im, G, adj=False):
    # im = im.to(config.DEVICE)
    G.eval()
    im_out = G(im.reshape([1, 3, 256, 256]))
    im_out = im_out[-1] * 0.5 + 0.5
    # im_out = F.adjust_saturation(im_out, 1.3)
    # im_out = F.adjust_hue(im_out, -0.002)
    # im_out = F.adjust_contrast(im_out, 1.02)
    return im_out


def use_model2(im, G, adj=False):
    # im = im.to(config.DEVICE)
    G.eval()
    im_out = G(im.reshape([1, 3, 256, 256]))
    im_out = im_out * 0.5 + 0.5
    # im_out = F.adjust_saturation(im_out, 1.3)
    # im_out = F.adjust_hue(im_out, -0.002)
    # im_out = F.adjust_contrast(im_out, 1.02)
    return im_out


def main():

    model_name = "model_0_gen.pth.tar"
    G_HE = load_model(os.path.join(config.CHECKPOINT_dir, model_name), G)  # pix2pixGAN

    G_HE.eval()

    data_source_dir = "data/test/"
    count = 0
    folder = "results/model_outputs/" # save folder
    if not os.path.exists(folder):
        os.makedirs(folder)
    for file in os.listdir(data_source_dir):
        im = Image.open(data_source_dir + file)
        im = np.array(im)
        # im = im / np.max(im)
        x = im[:, :256, :]
        y = im[:, 256:512, :]
        # print(type(x[0, 0, 0]))
        # x = block_reduce(x, block_size=(2, 2, 1), func=np.max)
        # y = block_reduce(y, block_size=(2, 2, 1), func=np.max)
        # print(type(x[0, 0, 0]))
        input_image = config.transform_only_input(image=x)["image"]
        mask = config.transform_only_mask(image=y)["image"]
        # input_image = torch.from_numpy(input_image)
        input_image = input_image.to(config.DEVICE)
        mask = mask.to(config.DEVICE)
        input_image = input_image.reshape([1, 3, 256, 256])
        mask = mask.reshape([1, 3, 256, 256])
        im_out_HE = use_model(input_image, G_HE)
        # im_out_HE_new_4 = use_model(input_image, G_HE_new_4)
        # im_out_HE_new_5 = use_model(input_image, G_HE_new_5)
        # im_out_HE = use_model2(input_image, G_HE)
        # im_out_H = use_model2(input_image, G_H)
        # im_out_E = use_model2(input_image, G_E)

        # im_out_H = im_out_H.reshape([3, 256, 256])
        # im_out_H = torch.sum(im_out_H, axis=0)
        # im_out_H = im_out_H / torch.max(im_out_H).item()
        # R = 1 - (1 - 0.5) * im_out_H
        # G = 1 - (1 - 0.54) * im_out_H
        # B = 1 - (1 - 1) * im_out_H
        # im_out_H_new = torch.stack([R, G, B], 0)
        # im_out_H_new = im_out_H_new.reshape([1, 3, 256, 256])
        #
        # im_out_E = im_out_E.reshape([3, 256, 256])
        # im_out_E = torch.sum(im_out_E, axis=0)
        # im_out_E = im_out_E / torch.max(im_out_E).item()
        # im_out_H = im_out_H / torch.max(im_out_H).item()
        # R_ = (1 - 0.5) * (-torch.log10(1-im_out_H + 1e-5)) + (1 - 1) * (-torch.log10(1-im_out_E + 1e-5))
        # G_ = (1 - 0.54) * (-torch.log10(1-im_out_H + 1e-5)) + (1 - 0.55) * (-torch.log10(1-im_out_E + 1e-5))
        # B_ = (1 - 1) * (-torch.log10(1-im_out_H + 1e-5)) + (1 - 0.88) * (-torch.log10(1-im_out_E + 1e-5))
        # im_out_HplusE = torch.stack([(10**(-R_)), (10**(-G_)), (10**(-B_))], 0)
        # im_out_HplusE = im_out_HplusE.reshape([1, 3, 256, 256])

        # scale = 0.55
        # weight = 0.9
        # # im_enhance = (im_out_HE + weight * im_out_H_new) / torch.max(im_out_HE + weight * im_out_H_new).item()
        # im_enhance = (im_out_HE + weight * im_out_H_new) * scale
        # im_enhance = F.adjust_saturation(im_enhance, 1.4)
        # im_enhance = F.adjust_hue(im_enhance, 0.005) #-.002
        # im_enhance = F.adjust_contrast(im_enhance, 1.02)

        # img = torch.cat([input_image * 0.5 + 0.5, im_out_HE, im_enhance, im_out_HE_new,
        #                  mask * 0.5 + 0.5], 0)
        img = torch.cat([input_image * 0.5 + 0.5, im_out_HE, mask * 0.5 + 0.5], 3)
        # img = torch.cat([im_out_HE_new_1, im_out_HE_new_2, im_out_HE_new_3, im_out_HE_new_4, im_out_HE_new_5, mask * 0.5 + 0.5], 3)
        save_image(img, folder + file)
        # gen.train()
        count += 1
        print("------" + str(count) + "/" + str(len(os.listdir(data_source_dir))) + "-------")


if __name__ == "__main__":
    main()

