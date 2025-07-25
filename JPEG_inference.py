import argparse
import copy
import json
import os
from os.path import join
import sys
import matplotlib.image
from tqdm import tqdm
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision import transforms

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


# init attacker
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


if __name__ == "__main__":
    args_attack = parse()
    print(args_attack)
    os.system(
        'cp -r ./results {}/results{}'.format(args_attack.global_settings.results_path, args_attack.attacks.momentum))
    print("experiment dir is created")
    os.system('cp ./setting.json {}'.format(os.path.join(args_attack.global_settings.results_path,
                                                         'results{}/setting.json'.format(
                                                             args_attack.attacks.momentum))))
    print("experiment config is saved")

    pgd_attack = init_Attack(args_attack)

    # load the trained JPEG-Watermark
    if args_attack.global_settings.universal_perturbation_path:
        pgd_attack.up = torch.load(args_attack.global_settings.universal_perturbation_path)

    # Init the attacked models
    # attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F_, T, G, E, reference, gen_models = prepare()
    # print("finished init the attacked models")
    attack_dataloader, test_dataloader, attgan, attgan_args, jpeg_solver, compression_model = prepare()
    # print("finished init the attacked models, only attack 2 epochs")
    # attack_dataloader, test_dataloader, attgan, attgan_args, jpeg_solver, transform, F, T, G, E, reference, gen_models, compression_model = prepare()

    tf = transforms.Compose([
        # transforms.CenterCrop(170),
        transforms.Resize(args_attack.global_settings.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image = Image.open(sys.argv[1])
    img = image.convert("RGB")
    img = tf(img).unsqueeze(0)

    # stargan inference and evaluating
    l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
    n_dist, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        img_a = img.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        x_noattack_list, x_fake_list, x_adv = jpeg_solver.test_universal_model_level(idx, img_a, c_org, pgd_attack.up,
                                                                         args_attack.stargan)
        out_adv_file = './demo_results/stargan_adv_original.jpeg'
        vutils.save_image(img_a.cpu(), out_adv_file,nrow=1, normalize=True, range=(-1, 1))
        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]
            l1_error += F.l1_loss(gen, gen_noattack)
            l2_error += F.mse_loss(gen, gen_noattack)
            l0_error += (gen - gen_noattack).norm(0)
            min_dist += (gen - gen_noattack).norm(float('-inf'))
            if F.mse_loss(gen, gen_noattack) > 0.05:
                n_dist += 1
            n_samples += 1

        ############# 保存图片做指标评测 #############
        # 保存原图
        out_file = './demo_results/stargan_original.jpg'

        vutils.save_image(img_a.cpu(), out_file, nrow=1, normalize=True, range=(-1., 1.))
        for j in range(len(x_fake_list)):
            # 保存原图生成图片
            gen_noattack = x_noattack_list[j]
            out_file = './demo_results/stargan_gen_{}.jpg'.format(j)
            vutils.save_image(gen_noattack, out_file, nrow=1, normalize=True, range=(-1., 1.))
            # 保存对抗样本生成图片
            gen = x_fake_list[j]
            out_file = './demo_results/stargan_advgen_{}.jpg'.format(j)
            vutils.save_image(gen, out_file, nrow=1, normalize=True, range=(-1., 1.))
        break
    print('stargan {} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(
        n_samples, l1_error / n_samples, l2_error / n_samples, float(n_dist) / n_samples, l0_error / n_samples,
        min_dist / n_samples))


