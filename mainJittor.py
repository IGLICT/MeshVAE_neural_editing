import argparse
import os
import numpy as np
import time
import cv2
from data import Data
from config import Config
from networkJittor import Model

import jittor as jt
from jittor import init
from jittor import nn
jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=['train', 'recon'])
parser.add_argument("--logfolder", type=str, default="temp", help="save checkpoint dir")
parser.add_argument("--dataset", type=str, default="scape", help="the training dataset name")

# ckp path
parser.add_argument("--cp_name",type=str, default="best.model", help="the checkpoint name")

# test input
parser.add_argument("--test_file",type=str, default="test.mat", help="the test file")

parser.add_argument("--lr",type=float, default=0.01,help="learning rate")
parser.add_argument("--epoch",type=int, default=30000, help="the training epoch")

# training loss weights
parser.add_argument("--lambda0",type=float, default=1,help="reconstruction loss")
parser.add_argument("--lambda1",type=float, default=1000,help="sparsity constraints")
parser.add_argument("--lambda2",type=float, default=1,help="weights norm loss")
parser.add_argument("--lambda3",type=float, default=1,help="trainable d loss")
parser.add_argument("--lambda4",type=float, default=0.01,help="KL loss")
parser.add_argument("--std",type=float, default=1,help="std")

# other network setting
parser.add_argument("--finaldim",type=int, default=9, help="the final layer dimension")
parser.add_argument("--latent",type=int, default=50, help="the latent dimension")
parser.add_argument("--K",type=int, default=3, help="the graph convolution parameter K")
parser.add_argument("--layer_num",type=int, default=1, help="number of convolution layers")
parser.add_argument("--th",type=int, default=10, help="the start valid threshold")

parser.add_argument("--seed",type=int, default=1, help="random seed")

#sparse constrain type
parser.add_argument("--d_type", type=str,  default='dynamic', choices=['fix', 'dynamic'], help='which sprase constrain to use')
# adjacency matrix type
parser.add_argument("--weight_type", type=str,  default='normal', choices=['normal', 'cot'], help='normal or cotangent adjacency matrix')
# convolution type
parser.add_argument("--conv_type", type=str,  default='spectral', choices=['spectral', 'spatial'], help='spectral or spatial convolution')
# activation function type
parser.add_argument("--ac_type", type=str,  default='tanh', choices=['none', 'tanh', 'selu'], help='actiation function type')
# network structure type
parser.add_argument("--net_type", type=str,  default='VAE', choices=['VAE', 'AE'], help='network structure type')
# synthesis input [component_id, max or min, weight]
parser.add_argument("--syn_list",nargs='+', type=int, default=[0,0,0], help='synthesis input')

parser.add_argument("--deform_weight", type=int, default=10, help='weight of defrom')
parser.add_argument("--deform_lr", type=float, default=0.01, help='weight of defrom')
parser.add_argument("--deform_epoch", type=int, default=1000, help='weight of defrom')

opt = parser.parse_args()
print(opt)
config = Config(opt)

data = Data(config)

model = Model(config, data)
optimizer = model.optimizer

from jittor.dataset import Dataset
class MyDataset(Dataset):
    def __init__(self, mydata):
        super().__init__()
        self.data = mydata

    def __getitem__(self, k):
        return self.data[k]

    def __len__(self,):
        return len(self.data)

# train_loader = MyDataset(data.train_data).set_attrs(batch_size=16, shuffle=True)
# train_loader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

# ----------
#  Training
# ----------

for epoch in range(config.epoch):
    # for i, (imgs, _) in enumerate(train_loader):
    # for i, train_input in enumerate(train_loader):
    sta = time.time()
    train_input = jt.array(data.train_data).stop_grad().astype(jt.float32)
    # train_input = jt.array(data.train_data).astype(jt.float32)
    # -----------------
    #  Train Generator
    # -----------------
    KL_loss, Generation_loss, laplacian_norm, weights_norm, dloss = model(train_input)
    # sumLoss = KL_loss + Generation_loss + laplacian_norm
    # sumLoss = KL_loss + Generation_loss
    sumLoss = Generation_loss + laplacian_norm + weights_norm + dloss

    optimizer.step(sumLoss)

    # ---------------------
    #  Train Discriminator
    # ---------------------

    print(
        "[Epoch %d/%d] [KL loss: %f] [Gen loss: %f] [Laplacian loss: %f] [weights_norm: %f] [dloss: %f] [Time: %f]"
        % (epoch, config.epoch, KL_loss.data, Generation_loss.data, laplacian_norm.data, weights_norm.data, dloss.data, time.time() - sta)
    )
    # print("Epoch %s has done ..." % epoch)