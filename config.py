
import os

class Config():
    
    start_ind={'face':346,'swing':135,'jump':135,'scape':36,'humanoida':138,'swingalign':135,'jumpalign':135,'horse':44,'fat2':400, 'fat': 846}
    end_ind={'face':385,'swing':150,'jump':150,'scape':72,'humanoida':154,'swingalign':150,'jumpalign':150,'horse':49, 'fat2':469, 'fat': 941}

    th={'face':10,'swing':40,'jump':100,'scape':15,'humanoid':1.5,'swingalign':50,'jumpalign':70,'horse':50, 'fat2':100, 'fat': 10}

    def __init__(self, FLAGS):
        self.latent = FLAGS.latent
        self.finaldim = FLAGS.finaldim
        self.epoch = FLAGS.epoch
        self.weight_type = FLAGS.weight_type
        self.layer_num = FLAGS.layer_num
        self.mode = FLAGS.mode
        self.cp_name = FLAGS.cp_name
        # self.test_file = FLAGS.test_file
        self.d_type = FLAGS.d_type
        self.lambda0 = FLAGS.lambda0
        self.lambda1 = FLAGS.lambda1
        self.lambda2 = FLAGS.lambda2
        self.lambda3 = FLAGS.lambda3
        self.lambda4 = FLAGS.lambda4
        self.logfolder = FLAGS.logfolder
        self.net_type = FLAGS.net_type
        self.seed = FLAGS.seed
        self.stddev = FLAGS.std
        dataname=FLAGS.dataset

        self.dataname = dataname
        self.featurefile='./data/'+dataname+'/feature.mat'
        self.neighbourfile='./data/'+dataname+'/neighbor.mat'
        self.distancefile='./data/'+dataname+'/geodesic.mat'
        self.start_idx=self.start_ind[dataname]
        self.end_idx=self.end_ind[dataname]
        if self.th.__contains__(dataname): 
            self.threshold=self.th[dataname]
        else:
            self.threshold = FLAGS.th
        self.K = FLAGS.K
        self.conv_type = FLAGS.conv_type
        self.ac_type=FLAGS.ac_type

        if self.weight_type=='normal':
            self.weightfile='./data/'+dataname+'/weight.mat'
        else:
            self.weightfile='./data/'+dataname+'/cotweight.mat'

        if not os.path.isdir('./checkpoint/'+self.logfolder) and self.mode=='train':
            os.mkdir('./checkpoint/'+self.logfolder)
        self.meshinfofile = './data/'+dataname+'/meshinfo.mat'

        syn_list = FLAGS.syn_list
        self.comp_idx = syn_list[0:int(len(syn_list)/3)]
        self.max_min = syn_list[int(len(syn_list)/3):int(len(syn_list)/3*2)]
        self.comp_weight = syn_list[int(len(syn_list)/3*2):]