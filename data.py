
import pickle
import scipy.io as sio
import numpy as np
import os
import scipy




class Data():
    def __init__(self, config):

        self.feature, self.logrmin, self.logrmax, self.smin, self.smax, self.pointnum = self.load_data(config.featurefile)
        if os.path.exists('./data/idx/'+config.dataname+'.dat'):
            split_idx = pickle.load(open('./data/idx/'+config.dataname+'.dat','rb'))
        else:
            split_idx = range(len(self.feature))

        start_idx = config.start_idx
        end_idx = config.end_idx


        train_data = self.feature
        valid_data = self.feature
        # if config.dataname == 'scape':
        #     valid_data = [self.feature[i] for i in split_idx[start_idx:end_idx]]
        #     train_data = [self.feature[i] for i in split_idx[0:start_idx]]
        # else:
        #     train_data = [self.feature[i] for i in split_idx[start_idx:end_idx]]
        #     valid_data = [self.feature[i] for i in split_idx[0:start_idx]]

        self.train_data = np.asarray(train_data,dtype='float32')
        self.valid_data = np.asarray(valid_data,dtype='float32')

        self.neighbour, self.degrees, self.maxdegree = self.load_neighbour(config.neighbourfile, self.pointnum)
        self.geodesic_weight = self.load_geodesic_weight(config.distancefile, self.pointnum)
        weight=self.load_weight(config.weightfile, self.pointnum)
        weight=weight.astype('float32')
        self.weight=scipy.sparse.csr_matrix(weight)
        # print(config.meshinfofile)
        # cc()
        self.load_meshinfo(config.meshinfofile)

    def load_meshinfo(self, path):
        meshinfo = sio.loadmat(path)
        self.face = meshinfo['f'].T
        self.vertices = meshinfo['v']
        self.vdiff = meshinfo['vdiff']
        self.recon = meshinfo['recon']


    def load_weight(self, path, pointnum, name='weight'):
        data = sio.loadmat(path)
        data = data[name]
        
        weight = np.zeros((pointnum,pointnum)).astype('float32')
        weight = data
        
        return weight
    def load_geodesic_weight(self, path, pointnum, name='point_geodesic'):

        
        data = sio.loadmat(path)
        data = data[name]
        
        distance = np.zeros((pointnum, pointnum)).astype('float32')
        distance = data

        return distance
    def load_neighbour(self, path, pointnum, name='neighbour'):
        data = sio.loadmat(path)
        data = data[name]
        maxdegree = data.shape[1]
        neighbour = np.zeros((pointnum, maxdegree)).astype('float32')
        neighbour = data
        degree = np.zeros((neighbour.shape[0], 1)).astype('float32')
        for i in range(neighbour.shape[0]):
            degree[i] = np.count_nonzero(neighbour[i])
        return neighbour,degree,maxdegree
    def load_data(self, path):
        resultmax = 0.95
        resultmin = -0.95
        
        data = sio.loadmat(path)
        logr = data['FLOGRNEW']
        s = data['FS']
        pointnum=logr.shape[1]
        logrmin = logr.min()
        logrmin = logrmin - 1e-6
        logrmax = logr.max()
        logrmax = logrmax + 1e-6
        smin = s.min()
        smin = smin- 1e-6
        smax = s.max()
        smax = smax + 1e-6
        
        rnew = (resultmax-resultmin)*(logr-logrmin)/(logrmax - logrmin) + resultmin
        snew = (resultmax-resultmin)*(s - smin)/(smax-smin) + resultmin
        
        feature = np.concatenate((rnew,snew),axis = 2)
        
        f = np.zeros_like(feature).astype('float32')
        f = feature
        
        return f, logrmin, logrmax, smin, smax,pointnum
    def load_test_data(self,path):
        resultmax = 0.95
        resultmin = -0.95
        
        data = sio.loadmat(path)
        logr = data['FLOGRNEW']
        s = data['FS']
        


        rnew = (resultmax-resultmin)*(logr-self.logrmin)/(self.logrmax - self.logrmin) + resultmin
        snew = (resultmax-resultmin)*(s - self.smin)/(self.smax-self.smin) + resultmin
        
        feature = np.concatenate((rnew,snew),axis = 2)
        
        f = np.zeros_like(feature).astype('float32')
        f = feature
        return f


    def recover_data(self, recover_feature, logrmin, logrmax, smin, smax, pointnum):
        logr = recover_feature[:,:,0:3]
        s = recover_feature[:,:,3:9]
        
        resultmax = 0.95
        resultmin = -0.95
        
        s = (smax - smin) * (s - resultmin) / (resultmax - resultmin) + smin
        logr = (logrmax - logrmin) * (logr - resultmin) / (resultmax - resultmin) + logrmin
        
        return s, logr