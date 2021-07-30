import jittor as jt
from jittor import nn
import os

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import utilsJittor as utilsJT

from jittor_geometric.nn import ChebConv

def Laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        #d += np.spacing(np.array(0.0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr_matrix
    return L
def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L

class Encoder(nn.Module):
    def __init__(self, input_dim, final_dim, layer_num):
        super(Encoder, self).__init__()
        # self.nb = nn.Linear(input_dim, final_dim)
        # self.edge = nn.Linear(input_dim, final_dim)
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, final_dim)
        self.norm = nn.BatchNorm1d(final_dim)
        self.action = nn.Tanh()

    def execute(self, input_feature):
        x = self.fc1(input_feature)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.action(x)
        return x

class SpatialConv(nn.Module):
    def __init__(self, input_dim, final_dim, nb, degrees):
        super(SpatialConv, self).__init__()
        self.nb = nb # [numV, 10]
        self.degrees = degrees # [numV, 1]
        self.nb_fc = nn.Linear(input_dim, final_dim)
        self.edge_fc = nn.Linear(input_dim, final_dim)
        self.norm = nn.BatchNorm1d(final_dim)
        self.action = nn.Tanh()
        self.numC = final_dim
        self.gather = utilsJT.SYTgather()

    def execute(self, input_feature): # [b,Vxc]
        bs = input_feature.shape[0]
        numV = input_feature.shape[1] // self.numC
        numC = self.numC
        input_feature = input_feature.reshape(bs, numV, numC)
        # padding 
        input_feature_ = jt.contrib.concat([jt.zeros([bs, 1, numC]), input_feature], dim=1) # # [b,V+1,c]
        
        # index_gather = self.nb.unsqueeze(0).unsqueeze(-1).expand([bs, numV, 10, numC])
        # index_gather = jt.broadcast(self.nb, shape=(bs, numV, 10, numC), dims=[0,-1])

        # nb_feature = []
        # for idx in range(numV):
        #     nb_feature.append(jt.gather(input_feature_, dim = 1, index=index_gather[:,idx]).unsqueeze(1))
        # nb_feature = jt.contrib.concat(nb_feature, dim=1)

        nb_feature = self.gather(input_feature_, self.nb)

        # input_feature_gather = input_feature_.unsqueeze(1).expand([bs, numV, numV+1, numC])
        # input_feature_gather = jt.broadcast(input_feature_, shape=(bs, numV, numV+1, numC), dims=[1])
        # nb_feature = jt.gather(input_feature_gather, dim = 2, index=index_gather) # [bs,numV,10,numC]

        mean_nb_feature = nb_feature.sum(dim=2) / self.degrees # [bs,numV,numC]
        
        nb_feature = self.nb_fc(mean_nb_feature.reshape(bs*numV, -1))
        edge_feature = self.edge_fc(input_feature.reshape(bs*numV, -1))

        total_feature = edge_feature + nb_feature

        x = self.norm(total_feature)
        x = self.action(x)
        return x.reshape(bs, -1)

class SpectralConv(nn.Module):
    def __init__(self, input_dim, final_dim, K, nb):
        super(SpectralConv, self).__init__()
        self.conv = ChebConv(input_dim, final_dim, K=K)
        edge_index = utilsJT.genEdge(nb.data)
        self.edge_index = jt.array(edge_index).astype(jt.int32).stop_grad()
        self.numC = final_dim

    def execute(self, input_feature):
        bs = input_feature.shape[0]
        numV = input_feature.shape[1] // self.numC
        numC = self.numC
        input_feature = input_feature.reshape(bs, numV, numC) # [b,V,c]
        # output = [self.conv(x, self.edge_index) for x in input_feature] # [b,V,c], [2, numE]，不支持batch操作...
        # return jt.stack(output).reshape(bs, -1)
        output = self.conv(input_feature, self.edge_index) # [b,V,c], [2, numE]，支持batch操作...
        return output.reshape(bs, -1)


class Decoder(nn.Module):
    def __init__(self, input_dim, final_dim, layer_num):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(final_dim, 100)
        self.fc2 = nn.Linear(100, input_dim)
        self.norm = nn.BatchNorm1d(input_dim)
        self.action = nn.Tanh()

    def execute(self, input_feature):
        x = self.fc1(input_feature)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.action(x)
        return x


class Model(nn.Module):
    def __init__(self, config, data):
        self.global_step = 0
        self.pointnum = data.pointnum
        self.maxdegree = data.maxdegree
        self.L = Laplacian(data.weight)
        self.L = rescale_L(self.L, lmax=2)
        self.config = config
        self.data = data
        self.constant_b_min = 0.2
        self.constant_b_max = 0.4
        jt.set_seed(self.config.seed)

        self.inputdim = 9
        self.finaldim = config.finaldim
        self.layer_num = config.layer_num

        self.nb = jt.array(data.neighbour).astype(jt.int32).stop_grad() # [V,10]
        self.degrees = jt.array(data.degrees).astype(jt.float32).stop_grad() # [V,1]
        self.laplacian = jt.array(data.geodesic_weight).astype(jt.float32).stop_grad() # [V,V]

        # self.C = jt.zeros([self.pointnum*self.finaldim, self.config.latent]).astype(jt.float32)
        # self.Cstd = jt.zeros([self.pointnum*self.finaldim, self.config.latent]).astype(jt.float32)
        self.C = 0.02 * jt.randn([self.pointnum*self.finaldim, self.config.latent]).astype(jt.float32)
        self.Cstd = 0.02 * jt.randn([self.pointnum*self.finaldim, self.config.latent]).astype(jt.float32)  
        
        ############## fc
        # self.encoder = Encoder(self.pointnum*self.finaldim, self.pointnum*self.finaldim, self.layer_num)
        # self.decoder = Decoder(self.pointnum*self.finaldim, self.pointnum*self.finaldim, self.layer_num)

        ############# spatial conv
        # self.encoder = SpatialConv(self.finaldim, self.finaldim, self.nb, self.degrees)
        # self.decoder = SpatialConv(self.finaldim, self.finaldim, self.nb, self.degrees)

        ############# spectral conv
        self.encoder = SpectralConv(self.finaldim, self.finaldim, self.config.K, self.nb)
        self.decoder = SpectralConv(self.finaldim, self.finaldim, self.config.K, self.nb)

        self.b0 = jt.ones([self.config.latent,1]).astype(jt.float32) * 0.2
        self.binarize = utilsJT.binarize()

        self.printGrad = utilsJT.printGrad()

        self.criterionL2 = nn.MSELoss()

        ### optimizer
        parameters = self.encoder.parameters()
        parameters += [self.C, self.Cstd, self.b0]
        parameters += self.decoder.parameters()
        self.optimizer = nn.Adam(parameters, lr=0.001)

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = jt.log(2. * np.pi)
        return jt.sum(
            -.5 * ((sample - mean) ** 2. * jt.exp(-logvar) + logvar + log2pi),
            dim=raxis)

    def reparameterization(self, mu, logvar):
        std = jt.exp(logvar / 2)
        sampled_z = jt.array(np.random.normal(0, 1, (mu.shape[0], self.config.latent))).float32().stop_grad()
        z = sampled_z * std + mu
        return z

    def execute(self, input):
        bs = input.shape[0]
        input = input.reshape(bs,-1) # reshape for the fc encoder

        x = self.encoder(input) # [bs, Vx9]
        mu = jt.matmul(x, self.C) # [bs, V*feature] x [V*feature, latent]
        sigma = jt.matmul(x, self.Cstd)
        x = self.reparameterization(mu, sigma)
        logpz = self.log_normal_pdf(x, 0, 0)
        logqz_x = self.log_normal_pdf(x, mu, sigma)

        weights_norm = jt.max(jt.abs(mu), dim=0) - 5 # [latent]
        weights_norm = jt.mean(jt.maximum(weights_norm, 0)) # float
        weights_norm = self.config.lambda2 * weights_norm

        y = jt.matmul(x, self.C.transpose(1,0)) # [bs, V*feature]
        y = self.decoder(y)

        # KL_loss = -(logpz - logqz_x).mean()
        # copy from https://wiseodd.github.io/techblog/2017/01/24/vae-pytorch/ (kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var))
        KL_loss = 0.5 * jt.sum(jt.exp(sigma) + mu**2 - 1 - sigma) * self.config.lambda4
        Generation_loss = self.criterionL2(input, y) * self.config.lambda0
        # loss = KL_loss + Generation_loss + self.laplacian_norm

        fcparams_group = self.C.reshape(self.pointnum, self.finaldim, self.config.latent).permute(2, 0, 1) # [latentNum, pointNum, featureDim]
        selfdot = jt.sum(jt.pow(fcparams_group, 2.0), dim = 2) # [latentNum, pointNum]
        maxdimension = jt.argmax(selfdot, dim = 1)[0] # [latentNum]
        maxlaplacian = jt.gather(self.laplacian, dim=0, index=jt.array(maxdimension)) # [latentNum, pointNum]
        # laplacian_new = jt.min(jt.max(jt.divide(jt.add(maxlaplacian,-self.constant_b_min),self.constant_b_max-self.constant_b_min),0, keepdims=True),1,keepdims=True)

        ### static
        # laplacian_new = jt.clamp((maxlaplacian-self.constant_b_min)/(self.constant_b_max-self.constant_b_min), 0, 1)
        ### dynamic
        # print("b0:", self.b0, self.b0.is_stop_grad())

        # self.b00 = self.printGrad(self.b0)
        # maxlaplacian = self.printGrad(maxlaplacian)

        laplacian_new = jt.maximum(self.binarize(maxlaplacian-self.b0), 0)

        laplacian_norm = self.config.lambda1 * jt.mean(jt.sum(jt.sqrt(selfdot) * laplacian_new, 1))

        dloss=self.config.lambda3 * (jt.sum(jt.maximum(self.b0,0)))
        # laplacian_norm = 0

        return KL_loss, Generation_loss, laplacian_norm, weights_norm, dloss
