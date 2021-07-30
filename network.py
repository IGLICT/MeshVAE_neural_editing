
import tensorflow as tf
from utils import *
from data import *
from f2v import Feature2Vertex
import os



class Model():
    def __init__(self, config, data):
        self.global_step = tf.Variable(
                initial_value = 0,
                name = 'global_step',
                trainable = False)
        self.pointnum = data.pointnum
        
        self.maxdegree = data.maxdegree
        self.L = Laplacian(data.weight)
        self.L = rescale_L(self.L, lmax=2)
        self.config = config
        self.data = data
        self.f2v = Feature2Vertex(self.data)
        constant_b_min = 0.2
        constant_b_max = 0.4
        tf.set_random_seed(self.config.seed)
        self.train_input = tf.constant(data.train_data,dtype = 'float32',name='train_input')
        self.valid_input = tf.constant(data.valid_data,dtype = 'float32',name='valid_input')
        self.KL_weights = tf.placeholder(tf.float32, (), name = 'KL_weights')
        self.inputs = tf.placeholder(tf.float32, [None, self.pointnum, 9], name = 'input_mesh')
        self.random_input = tf.placeholder(tf.float32, [None, self.config.latent], name='random_input')
        self.nb = tf.constant(data.neighbour, dtype='int32', shape=[self.pointnum, self.maxdegree], name='nb_relation')
        self.degrees = tf.constant(data.degrees, dtype = 'float32', shape=[self.pointnum, 1], name = 'degrees')
        self.embedding_inputs = tf.placeholder(tf.float32, [None, self.config.latent], name = 'embedding_inputs')
        self.laplacian = tf.constant(data.geodesic_weight,dtype = 'float32', shape =(self.pointnum, self.pointnum), name = 'geodesic_weight')
        self.finaldim = config.finaldim
        self.layer_num = config.layer_num
        for i in range(self.layer_num):
            if self.config.conv_type=='spatial':
                n, e = self.get_conv_weights(9, self.finaldim, name = 'convw'+str(i))
                setattr(self, 'n'+str(i+1), n)
                setattr(self, 'e'+str(i+1), e)
            else:
                setattr(self, 'W'+str(i*2+1), tf.get_variable("conv_weight"+str(i*2+1),[9*self.config.K, self.finaldim], tf.float32, tf.random_normal_initializer(stddev=0.02, seed = self.config.seed)))
                # setattr(self, 'W'+str(i*2+2), tf.get_variable("conv_weight"+str(i*2+2),[self.finaldim*self.config.K, 9], tf.float32, tf.random_normal_initializer(stddev=0.02, seed = self.config.seed)))

        if self.config.d_type=='dynamic':
            self.b0=tf.get_variable("b0",[self.config.latent,1],tf.float32,tf.constant_initializer(0.2))
        self.fcparams = tf.get_variable("weights", [self.pointnum*self.finaldim, self.config.latent], tf.float32, tf.random_normal_initializer(stddev=0.02, seed = self.config.seed))
        self.stdparams = tf.get_variable("stdweights", [self.pointnum*self.finaldim, self.config.latent], tf.float32, tf.random_normal_initializer(stddev=0.02, seed = self.config.seed))
        self.fcparams_group = tf.transpose(tf.reshape(self.fcparams, [self.pointnum, self.finaldim, self.config.latent]), perm = [2, 0, 1])

        self.selfdot = tf.reduce_sum(tf.pow(self.fcparams_group, 2.0), axis = 2)
        self.maxdimension = tf.argmax(self.selfdot, axis = 1)
        if self.config.dataname =='fat':
            self.maxdimension = [5459, 5712, 5467, 5717,  5016, 5048 ,5074, 5060, 4880, 4889, 4912 ,5365, 2000 ,2081, 2008, 2252 , 1691, 1602, 1721 ,1595 ,1505 ,1830 ,1389 ,1429  ,3511, 3050, 335 , 414 , 6765, 4589, 4393, 4985 , 3365, 1103, 907 , 1512 , 6786 ,4594 ,4931, 6513, 3387 ,1108 ,989  ,1246,  3380, 1115 ,973, 1454, 3146 ,3500]
        elif self.config.dataname =='scape':
            self.maxdimension = [920,  975,  937,  880,   1406, 1328, 1354, 1362,  1651, 1583, 1742, 1627,  1034, 1084, 1093, 1060,  1377, 1292 ,1309, 1337,1668, 1606, 1544, 1735,  1399, 1877, 2110, 2156 , 198,  380,  651,  1020,  203 , 334 , 603 , 946,   202 , 390 , 607,  911,219,  372 , 614,  855 ,  190,  361 , 625 , 905,   762 , 1219]
        self.maxlaplacian = tf.gather(self.laplacian, self.maxdimension)
        if self.config.d_type=='fix':
            self.laplacian_new=tf.minimum(tf.maximum(tf.div(tf.add(self.maxlaplacian,-constant_b_min),constant_b_max-constant_b_min),0),1)
        else:
            self.laplacian_new=tf.maximum(binarize(self.maxlaplacian-self.b0),0)
            
        self.laplacian_norm = self.config.lambda1*tf.reduce_mean(tf.reduce_sum(tf.sqrt(self.selfdot) * self.laplacian_new, 1))
        
        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(
              -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
              axis=raxis)


        if self.config.net_type == 'VAE':
            self.mean, self.std, self.weights_norm, _ = self.encoder_vae(self.train_input, train = True)
            eps = tf.random.normal(self.mean.shape, stddev=self.config.stddev, seed = self.config.seed)
            self.decoder_input = self.mean + tf.exp(self.std*.5)*eps
            logpz = log_normal_pdf(self.decoder_input, 0., 0.)
            logqz_x = log_normal_pdf(self.decoder_input, self.mean, self.std)
            self.KL_loss = -tf.reduce_mean(logpz - logqz_x)
            # self.decoder_input = self.mean + self.std*eps
            # self.KL_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.mean) + tf.square(self.std) - tf.log(1e-8 + tf.square(self.std)) - 1, 1))
            self.test_mean,self.test_std = self.encoder_vae(self.valid_input, train = False)
            test_eps = tf.random.normal(self.test_mean.shape, stddev=self.config.stddev, seed = self.config.seed)
            self.test_encode = self.test_mean + tf.exp(self.test_std*.5)*test_eps
            # self.test_encode = self.test_mean + self.test_std * test_eps
            self.feed_encode,self.feed_std=self.encoder_vae(self.inputs, train = False)

            #self.test_KL_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.test_mean) + tf.square(self.test_std) - tf.log(1e-8 + tf.square(self.test_std)) - 1, 1))
        else:
            self.decoder_input, self.weights_norm, _ = self.encoder(self.train_input, train = True)
            self.KL_loss=tf.constant(0,dtype = 'float32',shape=[1],name='KL_loss')
            self.test_encode = self.encoder(self.valid_input, train = False)
            self.feed_encode=self.encoder(self.inputs, train = False)

        self.KL_loss = self.KL_loss * self.KL_weights
        # self.KL_loss = self.config.lambda4 * self.KL_loss
        self.weights_norm = self.config.lambda2 * self.weights_norm

        self.decode = self.decoder(self.decoder_input, train = True)
        self.test_decode = self.decoder(self.test_encode, train = False)

        # self.test_generation_loss = tf.reduce_sum(tf.pow(self.valid_input-self.test_decode, 2.0), [1,2])

        self.test_generation_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.valid_input-self.test_decode, 2.0), [1,2]))
        

        self.feed_decode = self.decoder(self.feed_encode,train = False)
        self.embedding_output=self.decoder(self.embedding_inputs,train = False)
        self.random_decoder = self.decoder(self.random_input, train=False)
        
        self.generation_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.train_input-self.decode, 2.0), [1,2]))


        if self.config.d_type=='dynamic':
            self.dloss=self.config.lambda3 * (tf.reduce_sum(tf.maximum(self.b0,0)))
        else:
            self.dloss=tf.constant(0,dtype = 'float32',shape=[1],name='dloss')

        self.loss = self.config.lambda0 * self.generation_loss + self.laplacian_norm + self.weights_norm + self.dloss + self.KL_loss

        
        

        learning_rate = tf.train.exponential_decay(
        0.001,
        self.global_step,
        decay_steps=4000,
        decay_rate=0.9,
        staircase=True)
        self.learning_rate = tf.maximum(learning_rate,1e-5)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = None)


    def spatial_conv(self, input_feature, input_dim, output_dim,nb_weights, edge_weights, name = 'meshconv', training = True, special_activation = False, no_activation = False, bn = True):
        with tf.variable_scope(name) as scope:
            
            padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, input_dim], tf.float32)
            
            padded_input = tf.concat([padding_feature, input_feature], 1)

            def compute_nb_feature(input_f):
                return tf.gather(input_f, self.nb)

            total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
            mean_nb_feature = tf.reduce_sum(total_nb_feature, axis = 2)/self.degrees
        
            # nb_weights = tf.get_variable("nb_weights", [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            nb_bias = tf.get_variable("nb_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            nb_feature = tf.tensordot(mean_nb_feature, nb_weights, [[2],[0]]) + nb_bias

            # edge_weights = tf.get_variable("edge_weights", [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            edge_bias = tf.get_variable("edge_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
            edge_feature = tf.tensordot(input_feature, edge_weights, [[2],[0]]) + edge_bias

            total_feature = edge_feature + nb_feature

            if bn == False:
                fb = total_feature
            else:
                fb = self.batch_norm_wrapper(total_feature, is_training = training)

            if no_activation == True:
                fa = fb
            elif special_activation == False:
                fa = self.leaky_relu(fb)
            else:
                fa = tf.tanh(fb)

            return fa

    def get_conv_weights(self, input_dim, output_dim, name = 'convweight'):
        with tf.variable_scope(name) as scope:
            n = tf.get_variable("nb_weights", [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02, seed = self.config.seed))
            e = tf.get_variable("edge_weights", [input_dim, output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02, seed = self.config.seed))

            return n, e

    def encoder_vae(self, input_feature, train = True):
        with tf.variable_scope("encoder_vae") as scope:
            if(train == False):
                scope.reuse_variables()
            conv = input_feature
            for i in range(self.layer_num):
                if self.config.conv_type=='spatial':
                    conv = self.spatial_conv(conv, 9, self.finaldim,getattr(self, 'n'+str(i+1)), getattr(self, 'e'+str(i+1)), name = 'conv'+str(i+1), special_activation = True, no_activation= False if i==self.layer_num-1 else True, training = train, bn = False)
                    
                else:

                    conv=spectral_conv(conv,self.L,self.finaldim,self.config.K,getattr(self,'W'+str(i*2+1)), name='conv' + str(i+1), activation = self.config.ac_type)
                  
                    
            l0 = tf.reshape(conv, [tf.shape(conv)[0], self.pointnum*self.finaldim])

            l1 = tf.matmul(l0, self.fcparams)
            std = tf.matmul(l0,self.stdparams)
            # std = 2*tf.sigmoid(tf.matmul(l0,self.stdparams))

            if train == True:
                weights_maximum = tf.reduce_max(tf.abs(l1), 0) - 5
                zeros = tf.zeros_like(weights_maximum)
                weights_norm = tf.reduce_mean(tf.maximum(weights_maximum, zeros))
                return l1,std, weights_norm,conv
            else:
                return l1,std
    def encoder(self, input_feature, train = True):
        with tf.variable_scope("encoder") as scope:
            if(train == False):
                scope.reuse_variables()
            conv = input_feature
            for i in range(self.layer_num):
                if self.config.conv_type=='spatial':
                    conv = self.spatial_conv(conv, 9, self.finaldim,getattr(self, 'n'+str(i+1)), getattr(self, 'e'+str(i+1)), name = 'conv'+str(i+1), special_activation = True, no_activation= False if i==self.layer_num-1 else True, training = train, bn = False)
                    
                else:
                    conv=spectral_conv(conv,self.L,self.finaldim,self.config.K,getattr(self,'W'+str(i*2+1)), name='conv' + str(i+1), activation = self.config.ac_type if i==self.layer_num-1 else 'none')

            l0 = tf.reshape(conv, [tf.shape(conv)[0], self.pointnum*self.finaldim])

            l1 = tf.matmul(l0, self.fcparams)
            

            if train == True:
                weights_maximum = tf.reduce_max(tf.abs(l1), 0) - 5
                zeros = tf.zeros_like(weights_maximum)
                weights_norm = tf.reduce_mean(tf.maximum(weights_maximum, zeros))
                return l1, weights_norm,conv
            else:
                return l1
            
    def decoder(self, latent_tensor, train = True):
        with tf.variable_scope("decoder") as scope:
            if(train == False):
                scope.reuse_variables()

            l1 = tf.matmul(latent_tensor, tf.transpose(self.fcparams))
            l2 = tf.reshape(l1, [tf.shape(l1)[0], self.pointnum, self.finaldim])
            conv = l2
            for i in range(self.layer_num):
                if self.config.conv_type=='spatial':
                    conv = self.spatial_conv(conv, self.finaldim, 9, tf.transpose(getattr(self, 'n'+str(i+1))), tf.transpose(getattr(self, 'e'+str(i+1))), name = 'conv'+str(i+1), training = train, special_activation = True, bn = False)
                else:
                    conv=spectral_conv(conv,self.L,9,self.config.K,getattr(self,'W'+str(i*2+1)),name='conv'+str(i+1), activation=self.config.ac_type)
        return conv
    
    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            file = open('./checkpoint/'+self.config.logfolder+'/script_result.txt', 'w')
            vfile = open('./checkpoint/'+self.config.logfolder+'/script_result_valid.txt', 'w')
            file2=open('./checkpoint/'+self.config.logfolder+'/b.txt', 'w')

            if os.path.exists('./checkpoint/'+self.config.logfolder+'/checkpoint'):
                self.saver.restore(sess, tf.train.latest_checkpoint('./checkpoint/'+self.config.logfolder))
                print('restore!')
            else:
                tf.global_variables_initializer().run()
            # x=sess.run(self.train_input)
            # print(sess.run(self.valid_input))
            # xx()

            global_step = sess.run(self.global_step)
            valid_best = float('inf')
            file.write("d_type:%s,lambda0:%f,lambda1:%f,lambda2:%f,lambda3:%f,lambda4:%f,k:%d,start_idx:%d,end_idx:%d" % (self.config.d_type, self.config.lambda0, self.config.lambda1, self.config.lambda2, self.config.lambda3, self.config.lambda4, self.config.K, self.config.start_idx, self.config.end_idx))

            for epoch in range(global_step,self.config.epoch):
                for step in range(1):
                    # KL_weights = 0
                    # if epoch >= 15000:
                    KL_weights = self.config.lambda4
                    cost_generation, cost_kl, cost_norm, cost_weights, cost_d, _, lr =sess.run([self.generation_loss, self.KL_loss, self.laplacian_norm, self.weights_norm, self.dloss, self.optimizer, self.learning_rate], feed_dict={self.KL_weights:KL_weights})

                    # x=sess.run(getattr(self, 'n1'))
                    # print(x)
                    # xx()
                    # if epoch > 15000 and epoch % 1000 == 0:
                    #     self.saver.save(sess, self.config.logfolder +'/'+str(epoch)+'.model')
                    cost_valid = sess.run(self.test_generation_loss)
                    # cost_valid = cost_valid[23]

                    print("Epoch: [%5d|%5d] lr: %.5f generation_loss: %.8f validation: %.8f norm_loss: %.8f weight_loss: %.8f dloss: %.8f klloss: %.8f" % (epoch+1,step+1,lr, cost_generation, cost_valid, cost_norm, cost_weights,cost_d, cost_kl))
                    file.write("Epoch: [%5d|%5d] lr: %.5f generation_loss: %.8f validation: %.8f norm_loss: %.8f weight_loss: %.8f dloss: %.8f klloss: %.8f\n" % (epoch+1,step+1,lr, cost_generation, cost_valid, cost_norm, cost_weights,cost_d,cost_kl))
                    # print(cost_valid)
                    # cc()
                    if cost_generation + cost_kl < 50:
                        if cost_generation + cost_kl < valid_best:
                            valid_best = cost_generation + cost_kl - 0.1
                            print('Save best!')
                            self.saver.save(sess, './checkpoint/'+self.config.logfolder +'/'+'best.model')
                            vfile.write("save best!\nEpoch: [%5d|%5d] generation_loss: %.8f, validation: %.8f, norm_loss: %.8f weight_loss: %.8f dloss: %.8f klloss: %.8f\n" % (epoch+1,step+1, cost_generation, cost_valid, cost_norm,cost_weights,cost_d, cost_kl))     
                            if self.config.d_type=='dynamic':
                                file2.write(str(self.b0.eval()))
                                file2.write('\n')   
                    else:
                        pass



    def export_obj(self, out, v, f):
        with open(out, 'w') as fout:
            for i in range(v.shape[0]):
                fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
            for i in range(f.shape[0]):
                fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    def show_embedding(self, logfolder, cp):
        savedir='./result/'+logfolder+'/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)
            embedding = sess.run(self.feed_encode, feed_dict = {self.inputs: self.data.feature})
            sio.savemat(savedir+'/embedding.mat', {'embedding':embedding})
            


    def deform(self, logfolder, cp, test_file, deform_weight=10, deform_lr = 0.01, total_epoch = 1000):
        # self.show_embedding(logfolder, cp)
        savedir='./result/'+logfolder+'/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if not os.path.exists(savedir+'/edit_'+test_file[:-4]):
            os.makedirs(savedir+'/edit_'+test_file[:-4])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            # run_metadata = tf.RunMetadata()
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)

            feature = self.data.load_test_data(test_file)
            data = sio.loadmat(test_file)
            control_point = tf.squeeze(tf.constant(data['cp'],dtype = 'int32',name='control_point'))
            other_point = np.setdiff1d(np.arange(self.data.pointnum),data['cp'])
            # other_point = tf.squeeze(tf.constant(np.setdiff1d(np.arange(self.data.pointnum),data['cp']),dtype = 'int32',name='other_point'))
            # print(other_point.shape)
            # cc()
            target_coor = tf.constant(data['target_coor'],dtype = 'float32',name='target_coor')
            embedding = sess.run(self.feed_encode, feed_dict = {self.inputs: feature})

            global_step = tf.Variable(
                initial_value = 0,
                name = 'edit_global_step',
                trainable = False)
            learning_rate = tf.train.exponential_decay(
                deform_lr,
                global_step,
                decay_steps=4000,
                decay_rate=0.9,
                staircase=True)

            embedding_inputs = tf.Variable(initial_value=embedding, name='embedding', shape = [1, self.config.latent], dtype='float32',trainable=True)
            recon_feature = self.decoder(embedding_inputs, train = False)
            T = self.f2v.ftoT(recon_feature, use_identity=True)


            recon_vertex = self.f2v.Ttov(T)
            temp = tf.pow(tf.gather(recon_vertex, control_point, axis = 1) - target_coor, 2.0)
            if len(temp.shape) == 2:
                edit_loss = tf.reduce_mean(tf.reduce_sum(temp, [1])) #* 100000
            else:
                edit_loss = tf.reduce_mean(tf.reduce_sum(temp, [1,2])) #* 10000

            edit_loss = deform_weight * edit_loss
            # print(edit_loss)
            # preserve_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.gather(recon_vertex, other_point, axis = 1) - self.data.vertices[other_point,:], 2.0), [1,2]))
            # print(feature[:,other_point,:].shape)
            # print(tf.gather(recon_feature, other_point, axis = 1).shape)
            preserve_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.gather(recon_feature, other_point, axis = 1) - feature[:,other_point,:], 2.0), [1,2]))
            edit_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # edit_optimizer = tf.train.AdamOptimizer(learning_rate)
            total_loss = edit_loss# + preserve_loss
            edit_optim = edit_optimizer.minimize(total_loss, var_list=[embedding_inputs], global_step = global_step)
            intial_op = tf.variables_initializer([global_step,embedding_inputs] + edit_optimizer.variables())
            sess.run(intial_op)
            # T = sess.run(T)
            # print(rf.shape)

            # total_epoch = 1000
            best = np.inf
            embedding_seq = []
            import time
            t1 = time.time()
            for epoch in range(total_epoch):
                edit_loss_print, preserve_loss_print,emb, lr, _ = sess.run([edit_loss,preserve_loss, embedding_inputs, learning_rate, edit_optim])

                # embedding = sess.run(embedding_inputs)
                # embedding_seq.append(embedding)
                # if edit_loss_print+preserve_loss_print < best:
                    # if edit_loss_print < best and preserve_loss_print < 30:
                    # best = (edit_loss_print+preserve_loss_print)*0.9
                    # best = (edit_loss_print)*0.9
                    # vert = sess.run(recon_vertex).squeeze()
                    # self.export_obj(savedir+'/'+str(epoch+1)+'_best.obj',vert, self.data.face)
                print("Epoch: [%5d] lr: %.5f edit_loss: %.8f preserve_loss: %.8f" % (epoch+1, lr, edit_loss_print, preserve_loss_print))
                # print(emb)
            t2 = time.time()
            print('time:',t2-t1)
            vert = sess.run(recon_vertex).squeeze()

            # vert2 = sess.run(recon_vertex2, feed_dict = {self.inputs: self.data.feature})
            # print(vert2[0])

            
            self.export_obj(savedir+'/edit_'+test_file[:-4]+'/'+str(deform_weight)+'_'+str(deform_lr)+'_'+str(total_epoch)+'.obj',vert, self.data.face)
            # sio.savemat(savedir+'/edit_'+test_file[:-4]+'/'+'/seq_emb'+str(deform_weight)+'_'+str(deform_lr)+'_'+str(total_epoch)+'.mat', {'embedding':np.array(embedding),'RS':rs, 'RLOGR':rlogr,'T':T,'R':R,'S':S,'sum_T':sum_T,'vert':vert,'bdiff':bdiff})

            # from tensorflow.python.client import timeline
            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open(savedir+'/edit_'+test_file[:-4]+'/'+'timeline.json', 'w') as f:
            #     f.write(ctf)


    def inter_from_embedding(self, logfolder, cp, test_file):
        savedir='./result/'+logfolder + '/inter/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)
            embedding = sio.loadmat(test_file)['embedding']
            inter_result = np.linspace(embedding[0], embedding[1], 10)
            recover = sess.run(self.feed_decode, feed_dict = {self.feed_encode:inter_result})
            rs, rlogr = self.data.recover_data(recover, self.data.logrmin, self.data.logrmax, self.data.smin, self.data.smax, self.data.pointnum)
            sio.savemat(savedir+'/inter_from_embedding.mat', {'RS':rs, 'RLOGR':rlogr})

    def inter_from_mat(self, logfolder, cp, test_file):
        savedir='./result/'+logfolder + '/inter/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)
            embedding = sess.run(self.feed_encode, feed_dict = {self.inputs: self.data.load_test_data(test_file)})
            inter_result = np.linspace(embedding[0], embedding[1], 10)
            recover = sess.run(self.feed_decode, feed_dict = {self.feed_encode:inter_result})
            rs, rlogr = self.data.recover_data(recover, self.data.logrmin, self.data.logrmax, self.data.smin, self.data.smax, self.data.pointnum)
            sio.savemat(savedir+'/inter_from_mat.mat', {'RS':rs, 'RLOGR':rlogr})

    def recover_test(self, logfolder, cp, test_file):
        savedir='./result/'+logfolder+'/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)
            recover = sess.run(self.feed_decode, feed_dict = {self.inputs: self.data.load_test_data(test_file)})
            rs, rlogr = self.data.recover_data(recover, self.data.logrmin, self.data.logrmax, self.data.smin, self.data.smax, self.data.pointnum)
            sio.savemat(savedir+'/test.mat', {'RS':rs, 'RLOGR':rlogr})

    def embedd_inter(self, logfolder, cp):
        savedir='./result/'+logfolder+'/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)
            embedding = sess.run(self.feed_encode, feed_dict = {self.inputs: self.data.feature})
            down_id = [8, 9, 50, 52, 56] - 1
            up_id = [14, 24, 30, 43] - 1
            for i in range(len(down_id)):
                for j in range(len(up_id)):
                    inter_result = np.linspace(embedding[down_id[i]], embedding[up_id[j]], 10)
                    diff = inter_result[1] - inter_result[0]
                    all_point = np.array([inter_result[0]-diff*k for k in range(5,0,-1)] + inter_result.tolist() + [inter_result[-1]+diff*k for k in range(1,6)])
                    # cc()
                    recover = sess.run(self.feed_decode, feed_dict = {self.feed_encode:all_point})
                    rs, rlogr = self.data.recover_data(recover, self.data.logrmin, self.data.logrmax, self.data.smin, self.data.smax, self.data.pointnum)
                    sio.savemat(savedir+'/'+str(down_id[i])+'_'+str(up_id[j])+'.mat', {'RS':rs, 'RLOGR':rlogr})


    def random_generation(self, logfolder, cp):
        savedir='./result/'+logfolder+'/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)
            embedding = sess.run(self.feed_encode, feed_dict = {self.inputs: self.data.feature})
            # for i in range(5):
            i=4
            scalar = 0.1+0.1*i
            random_input = embedding[68] + gaussian(128, self.config.latent, var = scalar)
            recover = sess.run(self.random_decoder, feed_dict = {self.random_input:random_input})
            rs, rlogr = self.data.recover_data(recover, self.data.logrmin, self.data.logrmax, self.data.smin, self.data.smax, self.data.pointnum)
            sio.savemat(savedir+'/random'+str(i)+'.mat', {'RS':rs, 'RLOGR':rlogr,'emb':random_input})

    def recover_mesh(self, logfolder, cp):
        savedir='./result/'+logfolder+'/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)
            if os.path.exists('./data/idx/'+self.config.dataname+'.dat'):
                split_idx = pickle.load(open('./data/idx/'+self.config.dataname+'.dat','rb'))
            else:
                split_idx=ss[self.config.dataname]
            # random_input = gaussian(len(feature), self.config.latent)
            recover, embedding = sess.run([self.feed_decode, self.feed_encode], feed_dict = {self.inputs: self.data.feature})
            rs, rlogr = self.data.recover_data(recover, self.data.logrmin, self.data.logrmax, self.data.smin, self.data.smax, self.data.pointnum)
            sio.savemat(savedir+'/recover.mat', {'RS':rs, 'RLOGR':rlogr,'x':split_idx, 'embedding':embedding})
        return

    def interpolate(self,logfolder, cp):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        savedir='./result/'+logfolder + '/inter/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)
            data = sio.loadmat(savedir+'comp_latent.mat')
            ref = data['ref'][0,:]
            comp = data['comp']
            new_latent=[]
            for k in range(0,50):
                for i in range(3,16,4):
                    step = i/10.0
                    new_latent.append(ref + (comp[k][0] - ref) * step)
                    new_latent.append(ref + (comp[k][1] - ref) * step)
            recover = sess.run(self.embedding_output, feed_dict = {self.embedding_inputs: np.squeeze(new_latent)})
            rs, rlogr = self.data.recover_data(recover, self.data.logrmin, self.data.logrmax, self.data.smin, self.data.smax, self.data.pointnum)
        sio.savemat(savedir+'inter.mat',{'RS':rs,'RLOGR':rlogr})

    def individual_dimension(self, logfolder, cp, first = 0):
        savedir='./result/'+logfolder+'/dimension/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)

            embedding = sess.run(self.feed_encode, feed_dict = {self.inputs: self.data.feature})
            min_embedding = np.amin(embedding, axis = 0)

            max_embedding = np.amax(embedding, axis = 0)


            def generate_embedding_input(min, max, dimension, rest):
                x = np.zeros((25, self.config.latent)).astype('float32')

                for idx in range(self.config.latent):
                    if idx == dimension:
                        x[:, idx] = np.linspace(min[idx], max[idx], num = 25)
                    else:
                        x[:, idx] = rest[idx]

                return x
            comp=[]
            for idx in range(self.config.latent):
                embedding_data = generate_embedding_input(min_embedding, max_embedding, idx, embedding[first, :])

                recover = sess.run(self.embedding_output, feed_dict = {self.embedding_inputs: embedding_data})
                comp.append(embedding_data[[0,24],:,])
                rs, rlogr = self.data.recover_data(recover, self.data.logrmin, self.data.logrmax, self.data.smin, self.data.smax, self.data.pointnum)
                sio.savemat(savedir+'dimension'+str(idx+1)+'.mat', {'RS':rs, 'RLOGR':rlogr,'embedding':embedding_data})
            if not os.path.exists('./result/'+logfolder+'/inter/'):
                os.makedirs('./result/'+logfolder+'/inter/')
            sio.savemat('./result/'+logfolder+'/inter/comp_latent.mat', {'comp':comp,'ref':embedding})

    def component_view(self, logfolder, cp):
        savedir='./result/'+logfolder+'/component/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)

            component = sess.run([self.selfdot])[0]

            sio.savemat(savedir+'component.mat',{'component':component})

    def synthesize(self,logfolder, cp, comp_idx, max_min, comp_weight):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        savedir='./result/'+logfolder + '/synthesize/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, './checkpoint/'+logfolder + '/' + cp)
            data = sio.loadmat('./result/'+logfolder+'/inter/'+'comp_latent.mat')
            ref = data['ref']
            comp = data['comp']
            latent = []
            new_latent=ref[0,:]
            idx=0
            for i,j in zip(comp_idx,max_min):
                print(i,j)
                new_latent[int(i)] = comp[int(i)][int(j)][int(i)]*float(comp_weight[idx])
                latent.append(comp[int(i)][int(j)]*float(comp_weight[idx]))
                idx = idx + 1
            latent.append(new_latent)
            recover = sess.run(self.embedding_output, feed_dict = {self.embedding_inputs: latent})
            rs, rlogr = self.data.recover_data(recover, self.data.logrmin, self.data.logrmax, self.data.smin, self.data.smax, self.data.pointnum)
        sio.savemat(savedir+'synthesize.mat',{'RS':rs,'RLOGR':rlogr})