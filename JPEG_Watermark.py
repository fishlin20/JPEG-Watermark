import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F

from AttGAN.data import check_attribute_conflict

from data import CelebA
import attacks

from model_data_prepare import prepare



class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    return args_attack



args_attack = parse()
print(args_attack)
os.system('cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
print("experiment dir is created")
os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path,
                                                     'results{}/setting.json'.format(args_attack.attacks.momentum))))
print("experiment config is saved")


# init the attacker
def init_Attack(args_attack):
    pgd_attack = attacks.LinfPGDAttack(model=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                       epsilon=args_attack.attacks.epsilon, k=args_attack.attacks.k,
                                       k_agg=args_attack.attacks.k_agg,
                                       a=args_attack.attacks.a, b=args_attack.attacks.b,
                                       star_factor=args_attack.attacks.star_factor,
                                       attention_factor=args_attack.attacks.attention_factor,
                                       att_factor=args_attack.attacks.att_factor,
                                       HiSD_factor=args_attack.attacks.HiSD_factor, args=args_attack.attacks)
    return pgd_attack


pgd_attack = init_Attack(args_attack)


# init the attacker models with DiffJPEG
attack_dataloader, test_dataloader, attgan, attgan_args, jpeg_solver, transform, F, T, G, E, reference, gen_models, compression_model = prepare()
print("finished init the attacked models, attack 1 epochs next")

for i in range(1):
    # JPEG-starGAN---
    for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
        if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
            break
        # if idx == 1:
        #     break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)

#         # JPEG-StarGAN
        jpeg_solver.test_universal_model_jpeg_attack(idx, img_a, c_org, pgd_attack)

        # JPEG-AttGAN
        # att_b_list = [att_a]
        # for i in range(attgan_args.n_attrs):
        #     tmp = att_a.clone()
        #     tmp[:, i] = 1 - tmp[:, i]
        #     tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
        #     att_b_list.append(tmp)
        #
        # for i, att_b in enumerate(att_b_list):
        #     att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
        #     if i > 0:
        #         att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int
        #     with torch.no_grad():
        #         gen_noattack = attgan.G(img_a, att_b_)
        #     x_adv, perturb = pgd_attack.universal_perturb_attgan_DiffJPEG(flag, img_a, att_b_, gen_noattack, attgan,
        #                                                                   compression_model)
        #
        # JPEG-HiSD
        # with torch.no_grad():
        #     c = E(img_a)
        #     c_trg = c
        #     s_trg = F(reference, 1)
        #     c_trg = T(c_trg, s_trg, 1)
        #     x_trg = G(c_trg)
        #     mask = abs(x_trg - img_a)
        #     mask = mask[0, 0, :, :] + mask[0, 1, :, :] + mask[0, 2, :, :]
        #     mask[mask > 0.5] = 1
        #     mask[mask < 0.5] = 0
        # pgd_attack.universal_perturb_HiSD_DiffJPEG(flag, img_a.cuda(), transform, F, T, G, E, reference, x_trg + 0.002, gen_models,
        #                                   mask, compression_model)



    torch.save(pgd_attack.up, args_attack.global_settings.universal_perturbation_path)
    print('save the JPEG-Watermark')

print('The size of JPEG-Watermark: ', pgd_attack.up.shape)
