import argparse
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

from data import CelebA
import attacks

from AttGAN.attgan import AttGAN
from AttGAN.data import check_attribute_conflict
from AttGAN.helpers import Progressbar
from AttGAN.utils import find_model
import AttGAN.attacks as attgan_attack # Attack of AttGan
from stargan.solver import Solver

from HiSD.inference import prepare_HiSD
from jpeg_solver import Solver_jpeg

from DiffJPEG.modules import compress_jpeg, decompress_jpeg
from DiffJPEG.utils import diff_round,quality_to_factor,y_table,c_table

class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack

# init AttGAN
def init_attGAN(args_attack):
    with open(join('./AttGAN/output', args_attack.AttGAN.attgan_experiment_name, 'setting.txt'), 'r') as f:
        args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    args.test_int = args_attack.AttGAN.attgan_test_int
    args.num_test = args_attack.global_settings.num_test
    args.gpu = args_attack.global_settings.gpu
    args.load_epoch = args_attack.AttGAN.attgan_load_epoch
    args.multi_gpu = args_attack.AttGAN.attgan_multi_gpu
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)
    attgan.load(find_model(join('./AttGAN/output', args.experiment_name, 'checkpoint'), args.load_epoch))
    attgan.eval()
    return attgan, args
    # return args


# init stargan with DiffJPEG
def init_stargan_JPEG(args_attack, test_dataloader, compression_model):
    return Solver_jpeg(celeba_loader=test_dataloader, rafd_loader=None, config=args_attack.stargan, compression_model=compression_model, config_attack=args_attack.attacks)

# init attack data
def init_attack_data(args_attack, attgan_args):
    test_dataset = CelebA(args_attack.global_settings.data_path, args_attack.global_settings.attr_path, args_attack.global_settings.img_size, 'test', attgan_args.attrs,args_attack.stargan.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=args_attack.global_settings.batch_size, num_workers=0,
        shuffle=False, drop_last=False
    )
    if args_attack.global_settings.num_test is None:
        print('Testing images:', len(test_dataset))
    else:
        print('Testing images:', min(len(test_dataset), args_attack.global_settings.num_test))
    return test_dataloader

# init inference data
def init_inference_data(args_attack, attgan_args):
    test_dataset = CelebA(args_attack.global_settings.data_path, args_attack.global_settings.attr_path, args_attack.global_settings.img_size, 'test', attgan_args.attrs,args_attack.stargan.selected_attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, num_workers=0,
        shuffle=False, drop_last=False
    )
    if args_attack.global_settings.num_test is None:
        print('Testing images:', len(test_dataset))
    else:
        print('Testing images:', min(len(test_dataset), args_attack.global_settings.num_test))
    return test_dataloader


#init DiffJPEG
def init_diffjpeg(args_attack):
    quality = args_attack.jpeg.quality
    compress_network = compress_jpeg(factor=quality_to_factor(quality))
    # compress_network = compress_network.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    decompress_network = decompress_jpeg(
        height=256,
        width=256,
        factor=quality_to_factor(quality))
    # decompress_network = decompress_network.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return compress_network, decompress_network


def prepare():
    # prepare deepfake models
    args_attack = parse()
    attgan, attgan_args = init_attGAN(args_attack)
    # attgan_args = init_attGAN(args_attack)
    attack_dataloader = init_attack_data(args_attack, attgan_args)
    test_dataloader = init_inference_data(args_attack, attgan_args)
    #compression model
    compression_model = init_diffjpeg(args_attack)

    # solver = init_stargan(args_attack, test_dataloader)
    # solver.restore_model(solver.test_iters)
    jpeg_solver = init_stargan_JPEG(args_attack, test_dataloader, compression_model)
    jpeg_solver.restore_model(jpeg_solver.test_iters)

    transform, F, T, G, E, reference, gen_models = prepare_HiSD()
    print("Finished deepfake models initialization!")
    # return attack_dataloader, test_dataloader,attgan, attgan_args, jpeg_solver, transform, F, T, G, E, reference, gen_models, compression_model
    return attack_dataloader, test_dataloader,attgan, attgan_args, jpeg_solver, compression_model

if __name__=="__main__":
    prepare()