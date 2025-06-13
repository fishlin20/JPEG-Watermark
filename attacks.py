import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
try:
    import defenses.smoothing as smoothing
except:
    import AttGAN.defenses.smoothing as smoothing
import torchmetrics
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics import PeakSignalNoiseRatio as psnr

class LinfPGDAttack(object):
    def __init__(self, alpha=0.2, model=None, device=None, epsilon=0.05, k=10, k_agg=15, a=0.01, b=0.01,
                 star_factor=0.3,
                 attention_factor=0.3, att_factor=2, HiSD_factor=1, feat=None, args=None):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.alpha = alpha
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.k_agg = k_agg
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        # self.loss_fn = torch.nn.L1Loss().to(device)  # substitute the loss function.
        self.device = device
        self.b = b
        # Feature-level attack? Which layer?
        self.feat = feat
        self.psnr = psnr(data_range=1.0).to(device)

        # PGD or I-FGSM?
        self.rand = True

        # Universal perturbation
        self.up = None
        self.att_up = None
        self.attention_up = None
        self.star_up = None
        self.momentum = args.momentum
        self.p = None
        self.pre = None
        self.Y_cumulative_gradient = None
        # factors to control models' weights
        self.star_factor = star_factor
        self.attention_factor = attention_factor
        self.att_factor = att_factor
        self.HiSD_factor = HiSD_factor

    def universal_perturb_attgan_DiffJPEG(self, flag, X_nat, X_att, y, attgan, compression_model):
        """
        Vanilla Attack.
        """
        epsilon = self.epsilon
        compress_network, decompress_network = compression_model
        freq_range = (0, 8)
        # iter_up = self.att_up
        # if flag == 1 :
        if self.rand:
            # X = X_nat.clone().detach_() + torch.tensor(
            #     np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
            X = X_nat.clone().detach()
            X = (X + 1) / 2
            Y_nat, cb_nat, cr_nat = compress_network(X)
            Y = Y_nat.clone().detach() + torch.tensor(
                np.random.uniform(-self.epsilon, self.epsilon, Y_nat.shape).astype('float32')).to(self.device)
            # Y = torch.clamp(Y_nat.clone().detach() + self.up, min=-epsilon, max=epsilon).to(self.device)
            cb = cb_nat.clone().detach()
            cr = cr_nat.clone().detach()
            # self.pre = self.up.clone().detach()
        else:
            X = X_nat.clone().detach()
            X = (X + 1) / 2
            Y_nat, cb_nat, cr_nat = compress_network(X)
            cb = cb_nat.clone().detach()
            cr = cr_nat.clone().detach()


        for i in range(self.k_agg):
            Y = Y.detach().requires_grad_(True)
            decompress_network.zero_grad()
            attgan.G.zero_grad()
            x_jpeg = decompress_network(Y, cb, cr)

            loss_L_k = self.psnr(x_jpeg.float(), X_nat.float())
            # loss_L_k = ssim(x_jpeg, X_nat)
            x_jpeg_mod = x_jpeg.clone()
            x_jpeg_mod = x_jpeg_mod * 2 - 1
            output = attgan.G(x_jpeg_mod, X_att)
            # Minus in the loss means "towards" and plus means "away from"
            # loss = self.loss_fn(output, y)
            distortion_loss = self.loss_fn(output, y)

            loss = distortion_loss + self.alpha*(1/loss_L_k)
            loss.backward()
            # grad = X.grad
            if Y.grad is not None:
                Y_grad = Y.grad
                Y = self.add_frequency_perturbation(Y, Y_grad, self.a, freq_range)
                Y = Y.detach()

            eta = torch.mean(
                torch.clamp(
                    (Y - Y_nat),
                    min=-self.epsilon,
                    max=self.epsilon
                ).detach(),
                dim=0
            )
            
            # Y = Y_nat + eta
            # self.p = eta
            # X_adv = X + self.a * grad.sign()
            if self.up is None:

                self.up = eta
            else:

                self.up = self.up * self.momentum + eta * (1 - self.momentum)

            Y = torch.clamp(Y_nat + self.up, min=-epsilon, max=epsilon).detach()
            # self.up = self.up + self.b * self.pre

        X_adv = decompress_network(Y_nat + self.up, cb, cr)
        X = torch.clamp(X_adv, min=-1, max=1).detach()

        decompress_network.zero_grad()
        attgan.G.zero_grad()
        return X, X - X_nat


    def universal_perturb_DiffJPEG(self, X_nat, y, c_trg, model, compress_network, decompress_network):

        epsilon = self.epsilon

        Y_cumulative_grad = None
        freq_range = (0, 8)  # 设置频率范围 (低频)，可以修改为不同范围
        # iter_up = self.attention_up
        if self.rand:
            X = X_nat.clone().detach()
            X = (X + 1) / 2

            Y_nat, cb_nat, cr_nat = compress_network(X)  # X (0, 1)
            Y = Y_nat.clone().detach() + torch.tensor(np.random.uniform(-epsilon, epsilon, Y_nat.shape).astype(
                'float32')).to(self.device)
            cb = cb_nat.clone().detach()
            cr = cr_nat.clone().detach()
        else:
            X = X_nat.clone().detach()
            X = (X + 1) / 2
            Y_nat, cb_nat, cr_nat = compress_network(X)
            cb = cb_nat.clone().detach()
            cr = cr_nat.clone().detach()

        for i in range(self.k):
            Y = Y.detach().requires_grad_(True)
            decompress_network.zero_grad()
            model.zero_grad()
            x_jpeg = decompress_network(Y, cb, cr)

            loss_L_k = self.psnr(x_jpeg.float(), X_nat.float())
            # loss_L_k = ssim(x_jpeg, X_nat)
            x_jpeg_mod = x_jpeg.clone()
            # if error check：x_jpeg_mode[0] = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(x_jpeg_mode[0])
            x_jpeg_mod = x_jpeg_mod * 2 - 1
            output, feats = model(x_jpeg_mod, c_trg)
            if self.feat:
                output = feats[self.feat]

            distortion_loss = self.loss_fn(output, y)
            # Our method
            loss = distortion_loss + self.alpha * (1/loss_L_k)
            # loss = distortion_loss + self.alpha * loss_L_k
            # 测试没有PSNR损失
            # loss = distortion_loss
            loss.backward()
            # PGD att
            # if Y.grad is not None:
            #     Y = Y + self.a * Y.grad.sign()
            #     Y = Y.detach()
            # 应用频率掩码来控制扰动施加的频率范围
            if Y.grad is not None:
                Y_grad = Y.grad
                Y = self.add_frequency_perturbation(Y, Y_grad, self.a, freq_range)
                Y = Y.detach()

            eta = torch.mean(
                torch.clamp(
                    (Y - Y_nat),
                    min=-self.epsilon,
                    max=self.epsilon
                ).detach(),
                dim=0
            )

            if self.up is None:
                self.up = eta
            else:
                self.up = self.up * self.momentum + (1-self.momentum)*eta
            Y = torch.clamp(Y_nat + self.up, min=-epsilon, max=epsilon).detach()
            X_adv = decompress_network(Y_nat + self.up, cb, cr)
            X = torch.clamp(X_adv, min=-1, max=1).detach()
            # X = torch.clamp(X_nat + self.up, min=-1, max=1).detach()
            # print(loss.item())

        decompress_network.zero_grad()
        model.zero_grad()
        return X, X - X_nat

    def add_frequency_perturbation(self, Y, Y_grad, a, freq_range):
        """
        freq
        """
        Y_perturbed = Y.clone()
        h, w = Y_perturbed.shape[-2], Y_perturbed.shape[-1]

        # 创建与 Y 匹配的掩码
        mask = torch.zeros((h, w), device=Y.device)
        mask[freq_range[0]:freq_range[1], freq_range[0]:freq_range[1]] = 1
        mask = mask.unsqueeze(0).expand_as(Y)  # 扩展 mask 使其形状与 Y 相同

        # 只在选定的频率区域内施加扰动
        Y_perturbed = Y_perturbed + a * Y_grad.sign() * mask
        return Y_perturbed



    def universal_perturb_HiSD_DiffJPEG(self, flag, X_nat, transform, F, T, G, E, reference, y, gen, mask,
                                        compression_model):

        epsilon = self.epsilon
        freq_range = (0, 8)
        flag = flag
        compress_network, decompress_network = compression_model
        # if flag ==1:
        # if self.rand and self.up is not None:
        if self.rand:
            # X = X_nat.clone().detach() + self.up  # + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).to(self.device)
            X = X_nat.clone().detach()
            X = (X + 1) / 2
            Y_nat, cb_nat, cr_nat = compress_network(X)
            # Y = torch.clamp(Y_nat.clone().detach() + self.up, min=-epsilon, max=epsilon).to(self.device)
            Y = Y_nat.clone().detach() + torch.tensor(np.random.uniform(-epsilon, epsilon, Y_nat.shape).astype(
                'float32')).to(self.device)
            # Y = Y_nat.clone().detach() + self.up
            cb = cb_nat.clone().detach()
            cr = cr_nat.clone().detach()
            # self.pre = self.up.clone().detach()
        else:
            X = X_nat.clone().detach()
            X = (X + 1) / 2
            Y_nat, cb_nat, cr_nat = compress_network(X)
            Y = Y_nat.clone().detach()
            cb = cb_nat.clone().detach()
            cr = cr_nat.clone().detach()
        # else:
        #     X = X_nat.clone().detach()
        #     X = (X + 1) / 2
        #     Y_nat, cb_nat, cr_nat = compress_network(X)
        #     Y = torch.clamp(Y_nat.clone().detach() + self.up, min=-epsilon, max=epsilon).to(self.device)
        #     cb = cb_nat.clone().detach()
        #     cr = cr_nat.clone().detach()

        for i in range(self.k):
            # X.requires_grad = True
            Y = Y.detach().requires_grad_(True)
            decompress_network.zero_grad()
            gen.zero_grad()
            x_jpeg = decompress_network(Y, cb, cr)
            # loss_L_k = self.loss_fn(x_jpeg, X_nat)
            loss_L_k = self.psnr(x_jpeg.float(), X_nat.float())
            x_jpeg_mod = x_jpeg.clone()
            x_jpeg_mod = x_jpeg_mod * 2 - 1
            c = E(x_jpeg_mod)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            x_trg = G(c_trg)
            # model.zero_grad()
            # gen.zero_grad()
            distortion_loss = self.loss_fn(x_trg, y)
            # loss = self.loss_fn(x_trg, y)
            # loss = self.alpha * loss_L_k + distortion_loss
            loss = distortion_loss + self.alpha * (1/loss_L_k)
            loss.backward()

            # X_adv = X + self.a * grad.sign()
            if Y.grad is not None:
                Y_grad = Y.grad
                Y = self.add_frequency_perturbation(Y, Y_grad, self.a, freq_range)
                Y = Y.detach()

            eta = torch.mean(
                torch.clamp(
                    # self.HiSD_factor * (Y - Y_nat),
                    (Y - Y_nat),
                    min=-self.epsilon,
                    max=self.epsilon
                ),
                dim=0
            ).detach()
            # self.p = eta
            # Y = Y_nat + self.p
            if self.up is None:
                self.up = eta
            else:
                self.up = self.up * self.momentum + eta * (1 - self.momentum)

            Y = torch.clamp(Y_nat + self.up, min=-epsilon, max=epsilon).detach()

        X_adv = decompress_network(Y_nat + self.up, cb, cr)
        X = torch.clamp(X_adv, min=-1, max=1).detach()

        decompress_network.zero_grad()
        gen.zero_grad()
        return X, X - X_nat


def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res


def perturb_batch(X, y, c_trg, model, adversary):
    # Perturb batch function for adversarial training
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()

    adversary.model = model_cp

    X_adv, _ = adversary.perturb(X, y, c_trg)

    return X_adv
